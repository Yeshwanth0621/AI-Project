"""
ML Service for Worker Recommendation API
FastAPI endpoint for worker matching and recommendations
Includes chat interface, appointment booking, and task assignment with notifications
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
from pathlib import Path
from datetime import datetime
import json

# Add ml directory to path (parent directory now)
ml_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ml_dir))

from hybrid_recommender import HybridWorkerRecommender

# Import task database and notification service
from task_db import (
    init_db, create_task, get_task, get_worker_tasks, get_all_tasks,
    update_task_status, get_worker_contact, upsert_worker_contact,
    log_notification, get_task_stats
)
from notification_service import get_notification_service


class WorkerQuery(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "High voltage electrician with NFPA certification",
                "top_k": 5,
                "filters": {
                    "min_experience": 10,
                    "min_safety": 4.0,
                    "only_available": True
                }
            }
        }


class ChatMessage(BaseModel):
    message: str
    history: Optional[List[Dict]] = None
    context: Optional[Dict] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "i need an electrican for high voltage work",
                "history": [],
                "context": {"last_query": "electrician"}  # Example context
            }
        }


class AppointmentRequest(BaseModel):
    worker_id: int
    worker_name: str
    job_title: str
    job_description: str
    start_date: str
    duration: str
    location: str
    requester_name: str
    requester_email: str
    requester_phone: Optional[str] = None
    special_requirements: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": 15,
                "worker_name": "Alexandria Smith",
                "job_title": "Electrical Maintenance",
                "job_description": "High voltage maintenance work",
                "start_date": "2025-01-15",
                "duration": "2 weeks",
                "location": "Site A - Building 3",
                "requester_name": "John Doe",
                "requester_email": "john@company.com",
                "requester_phone": "+1234567890",
                "special_requirements": "NFPA compliance required"
            }
        }


class WorkerRecommendation(BaseModel):
    id: int
    name: str
    role: str
    experience: int
    safety_rating: float
    availability: str
    skills: str
    similarity_score: float
    suitability_score: float
    reasoning: str


# Initialize FastAPI app
app = FastAPI(
    title="Worker Recommendation API",
    description="API for AI-powered worker matching, chat interface, and appointment booking",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance (loaded once)
worker_recommender: Optional[HybridWorkerRecommender] = None

# Simple in-memory appointment storage (in production, use a database)
appointments_store: List[Dict] = []


@app.on_event("startup")
async def load_model():
    """Load worker recommender and initialize database on startup"""
    global worker_recommender
    
    # Initialize task database
    init_db()
    print("✅ Task database initialized!")
    
    # Change to ml directory for file access
    import os
    original_dir = os.getcwd()
    os.chdir(ml_dir)
    
    # Load worker recommender
    try:
        print("🚀 Loading Worker Recommender...")
        worker_recommender = HybridWorkerRecommender(use_slm=False)
        print("✅ Worker Recommender loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading worker recommender: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("⚠️  Worker recommendations will fail. Run: python prepare_worker_data.py && python train_decision_tree.py")
        worker_recommender = None
    
    # Change back to original directory
    os.chdir(original_dir)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Worker Recommendation API",
        "recommender_loaded": worker_recommender is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if worker_recommender is not None else "recommender_not_loaded",
        "recommender_ready": worker_recommender is not None,
        "message": "✅ Ready for recommendations" if worker_recommender else "⚠️ Recommender not loaded. Please run data preparation first."
    }


@app.post("/recommend-workers")
async def recommend_workers(request: WorkerQuery):
    """
    Recommend workers based on job requirements
    
    Args:
        request: WorkerQuery with query string, top_k, and optional filters
        
    Returns:
        List of worker recommendations with scores and reasoning
    """
    if worker_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Worker recommender not loaded. Please run data preparation and train decision tree."
        )
    
    try:
        # Get recommendations
        recommendations = worker_recommender.recommend(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        return {
            "query": request.query,
            "total_results": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Worker recommendation failed: {str(e)}"
        )


@app.post("/chat")
async def chat_with_assistant(request: ChatMessage):
    """
    Conversational chat endpoint for worker recommendations.
    Returns exactly 2 best workers with detailed reasoning.
    Handles typos, natural language queries, and CONTEXT.
    
    Args:
        request: ChatMessage with user message and optional history/context
        
    Returns:
        Chat response with worker recommendations and updated context
    """
    if worker_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Worker recommender not loaded. Please run data preparation and train decision tree."
        )
    
    try:
        user_message = request.message.strip()
        current_context = request.context or {}
        last_query = current_context.get("last_query", "")
        
        # --- Context Merging Logic ---
        final_query = user_message
        
        # Heuristic: Check for continuation words
        continuation_markers = ["also", "and", "plus", "with", "too", "additionally"]
        
        should_merge = False
        message_lower = user_message.lower()
        
        # 1. Check for explicit continuation markers
        if any(marker in message_lower.split() for marker in continuation_markers):
            should_merge = True
            
        # 2. Check for implicit continuation (short queries that might be adding a skill)
        # If the query is very short (e.g., "welding") and we have a previous query, assume adding on.
        elif len(message_lower.split()) <= 3 and last_query:
            # But avoid merging simple greetings or commands if we had those checks here
            should_merge = True

        if should_merge and last_query:
            final_query = f"{last_query} {user_message}"
            print(f"🔄 Merging context: '{last_query}' + '{user_message}' -> '{final_query}'")
        else:
            print(f"🆕 New context started: '{final_query}'")
            
        # Update context for next turn
        new_context = {"last_query": final_query}
        
        # --- End Context Merging Logic ---
        
        
        import random
        
        # Professional but friendly response variations
        empty_query_responses = [
            "Hello! I'm your AI Worker Allocation Assistant. Please describe the type of worker you need, including any specific skills or certifications required. For example: 'electrician with high voltage experience' or 'welder with TIG certification'.",
            "Welcome! I'm here to help you find qualified workers for your project. Tell me what role you're looking to fill and any specific requirements, and I'll find the best matches.",
            "Hello! Ready to help you find the right professional for your needs. Please describe the position and any required skills or experience."
        ]
        
        greeting_responses = [
            "Hello! How can I assist you today? Please describe the type of worker you're looking for, and I'll find the best candidates for you.",
            "Welcome! I'm your AI assistant for worker allocation. What kind of professional are you looking for today?",
            "Hello! I'm here to help match you with qualified workers. What position are you looking to fill?",
            "Hi there! Tell me about the role you need to fill, and I'll recommend the most suitable candidates."
        ]
        
        no_results_responses = [
            f"I wasn't able to find any available workers matching '{final_query}' at the moment. Could you try different keywords or broaden your search criteria?",
            f"Unfortunately, no workers matching '{final_query}' are currently available. Please try adjusting your requirements or describing the role differently.",
            f"I couldn't locate anyone matching '{final_query}' in our system. Consider using different skill keywords or a broader job description."
        ]
        
        intro_variations = [
            "I found some excellent candidates for you:\n\n",
            "Here are the top recommendations based on your requirements:\n\n",
            "Based on your criteria, I recommend the following professionals:\n\n",
            "I've identified the best matches for your needs:\n\n"
        ]
        
        # Generate conversational response based on message type
        if not final_query:
            return {
                "response": random.choice(empty_query_responses),
                "workers": [],
                "has_recommendations": False,
                "context": new_context
            }
        
        # Check for greetings (only if NOT merging - e.g. fresh start)
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'sup', 'yo', 'hola']
        if user_message.lower().strip() in greetings and not should_merge:
            return {
                "response": random.choice(greeting_responses),
                "workers": [],
                "has_recommendations": False,
                "context": new_context
            }
        
        # --- NEW: Check for ID-based or name-based search ---
        import re
        recommendations = []
        
        # Check for ID search: "id 15", "worker 15", "#15", "ID:15"
        id_match = re.search(r'(?:id|worker|#|id:)\s*(\d+)', final_query.lower())
        if id_match:
            worker_id = int(id_match.group(1))
            worker_df = worker_recommender.workers_df
            worker_row = worker_df[worker_df['ID'] == worker_id]
            if not worker_row.empty:
                w = worker_row.iloc[0]
                recommendations = [{
                    'id': int(w['ID']),
                    'name': w['Name'],
                    'role': w['Job_Role'],
                    'experience': int(w['Experience_Years']),
                    'safety_rating': float(w['Safety_Rating']),
                    'availability': w['Availability'],
                    'skills': w['Skills_Description'],
                    'similarity_score': 1.0,
                    'suitability_score': 1.0,
                    'reasoning': f"Direct lookup by ID {worker_id}. {w['Skills_Description'][:100]}..."
                }]
                print(f"🔍 Found worker by ID: {worker_id}")
        
        # Check for name search if no ID match
        if not recommendations:
            worker_df = worker_recommender.workers_df
            # Search by name (case insensitive partial match)
            name_matches = worker_df[worker_df['Name'].str.lower().str.contains(final_query.lower(), na=False)]
            if not name_matches.empty:
                for _, w in name_matches.head(2).iterrows():
                    recommendations.append({
                        'id': int(w['ID']),
                        'name': w['Name'],
                        'role': w['Job_Role'],
                        'experience': int(w['Experience_Years']),
                        'safety_rating': float(w['Safety_Rating']),
                        'availability': w['Availability'],
                        'skills': w['Skills_Description'],
                        'similarity_score': 1.0,
                        'suitability_score': 1.0,
                        'reasoning': f"Name match for '{final_query}'. {w['Skills_Description'][:100]}..."
                    })
                print(f"🔍 Found {len(recommendations)} worker(s) by name")
        
        # Fall back to skill-based recommendation if no ID/name match
        if not recommendations:
            # Get exactly 2 recommendations for chat interface using the FINAL merged query
            recommendations = worker_recommender.recommend(
                query=final_query,
                top_k=2,
                filters={"only_available": True}  # Only show available workers in chat
            )
        
        if not recommendations:
            return {
                "response": random.choice(no_results_responses),
                "workers": [],
                "has_recommendations": False,
                "context": new_context
            }
        
        # Generate professional conversational response
        worker_names = [r['name'] for r in recommendations]
        
        response_text = random.choice(intro_variations)
        
        if should_merge:
            response_text = f"I've updated the search to include '{user_message}'.\n\n" + random.choice(intro_variations)
        
        for i, rec in enumerate(recommendations, 1):
            experience_text = f"{rec['experience']} years of experience" if rec['experience'] < 10 else f"Over {rec['experience']} years of experience"
            safety_text = "excellent" if rec['safety_rating'] >= 4.5 else "strong" if rec['safety_rating'] >= 4.0 else "good"
            
            response_text += f"**{rec['name']}** — {rec['role']}\n"
            response_text += f"• {experience_text} with {safety_text} safety rating ({rec['safety_rating']:.1f}/5)\n"
            response_text += f"• {rec['reasoning']}\n\n"
        
        # Professional ending
        endings = [
            "Please review the candidate details below to proceed with booking.",
            "Select a candidate below to schedule an appointment.",
            "Click 'Book This Worker' on any card to proceed with the appointment."
        ]
        response_text += random.choice(endings)
        
        return {
            "response": response_text,
            "workers": recommendations,
            "has_recommendations": True,
            "context": new_context 
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )


@app.get("/worker/{worker_id}")
async def get_worker(worker_id: int):
    """
    Get details of a specific worker by ID.
    Used for the appointment page to display worker information.
    
    Args:
        worker_id: Worker ID from the database
        
    Returns:
        Worker details including skills and ratings
    """
    if worker_recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Worker recommender not loaded."
        )
    
    try:
        # Find worker by ID
        worker_df = worker_recommender.workers_df
        worker_row = worker_df[worker_df['ID'] == worker_id]
        
        if worker_row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Worker with ID {worker_id} not found."
            )
        
        worker = worker_row.iloc[0]
        
        return {
            "id": int(worker['ID']),
            "name": worker['Name'],
            "role": worker['Job_Role'],
            "experience": int(worker['Experience_Years']),
            "safety_rating": float(worker['Safety_Rating']),
            "availability": worker['Availability'],
            "skills": worker['Skills_Description']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get worker details: {str(e)}"
        )


@app.post("/book-appointment")
async def book_appointment(request: AppointmentRequest):
    """
    Book an appointment with a worker.
    Stores the appointment and returns confirmation.
    
    Args:
        request: AppointmentRequest with all booking details
        
    Returns:
        Confirmation with appointment ID
    """
    try:
        # Generate appointment ID
        appointment_id = f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{request.worker_id}"
        
        # Create appointment record
        appointment = {
            "appointment_id": appointment_id,
            "worker_id": request.worker_id,
            "worker_name": request.worker_name,
            "job_title": request.job_title,
            "job_description": request.job_description,
            "start_date": request.start_date,
            "duration": request.duration,
            "location": request.location,
            "requester_name": request.requester_name,
            "requester_email": request.requester_email,
            "requester_phone": request.requester_phone,
            "special_requirements": request.special_requirements,
            "status": "confirmed",
            "created_at": datetime.now().isoformat()
        }
        
        # Store appointment (in-memory for demo)
        appointments_store.append(appointment)
        
        print(f"📅 New appointment created: {appointment_id}")
        print(f"   Worker: {request.worker_name} (ID: {request.worker_id})")
        print(f"   Job: {request.job_title}")
        print(f"   Date: {request.start_date}")
        
        return {
            "success": True,
            "message": f"Appointment successfully booked!",
            "appointment_id": appointment_id,
            "details": {
                "worker_name": request.worker_name,
                "job_title": request.job_title,
                "start_date": request.start_date,
                "location": request.location
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to book appointment: {str(e)}"
        )


@app.get("/appointments")
async def get_appointments():
    """Get all booked appointments"""
    return {
        "total": len(appointments_store),
        "appointments": appointments_store
    }


# ==================== TASK ASSIGNMENT ENDPOINTS ====================

class TaskAssignment(BaseModel):
    """Request model for task assignment"""
    worker_id: int
    worker_name: str
    job_title: str
    job_description: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    duration: Optional[str] = None
    assigned_by: str = "admin"
    requester_name: Optional[str] = None
    requester_email: Optional[str] = None
    requester_phone: Optional[str] = None
    special_requirements: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": 15,
                "worker_name": "Alexandria Smith",
                "job_title": "Electrical Maintenance",
                "job_description": "High voltage panel inspection",
                "location": "Site A, Building 3",
                "start_date": "2026-01-05",
                "duration": "2 days",
                "assigned_by": "admin"
            }
        }


class TaskStatusUpdate(BaseModel):
    """Request model for task status update"""
    status: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "accepted"  # pending, accepted, in_progress, completed, declined
            }
        }


class WorkerContactUpdate(BaseModel):
    """Request model for worker contact info update"""
    email: Optional[str] = None
    phone: Optional[str] = None
    notification_enabled: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "worker@example.com",
                "phone": "+1234567890",
                "notification_enabled": True
            }
        }


class PushSubscription(BaseModel):
    """Request model for push notification subscription"""
    worker_id: int
    subscription: Dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "worker_id": 15,
                "subscription": {
                    "endpoint": "https://fcm.googleapis.com/...",
                    "keys": {
                        "p256dh": "...",
                        "auth": "..."
                    }
                }
            }
        }


@app.post("/assign-task")
async def assign_task(request: TaskAssignment):
    """
    Assign a task to a worker and send notifications.
    
    Args:
        request: TaskAssignment with all task details
        
    Returns:
        Task confirmation with notification status
    """
    try:
        # Create task in database
        task = create_task(
            worker_id=request.worker_id,
            worker_name=request.worker_name,
            job_title=request.job_title,
            job_description=request.job_description,
            location=request.location,
            start_date=request.start_date,
            duration=request.duration,
            assigned_by=request.assigned_by,
            requester_name=request.requester_name,
            requester_email=request.requester_email,
            requester_phone=request.requester_phone,
            special_requirements=request.special_requirements
        )
        
        print(f"📋 Task created: {task['id']} for worker {request.worker_name}")
        
        # Get worker contact info for notifications
        worker_contact = get_worker_contact(request.worker_id)
        worker_email = worker_contact.get('email') if worker_contact else None
        push_subscription = worker_contact.get('push_subscription') if worker_contact else None
        
        # Send notifications
        notification_service = get_notification_service()
        notification_results = notification_service.send_task_notification(
            worker_name=request.worker_name,
            worker_email=worker_email,
            push_subscription=push_subscription,
            task_title=request.job_title,
            task_description=request.job_description,
            task_location=request.location,
            start_date=request.start_date,
            assigned_by=request.assigned_by
        )
        
        # Log notification attempts
        log_notification(
            task_id=task['id'],
            worker_id=request.worker_id,
            notification_type='email',
            status='success' if notification_results['email']['success'] else 'failed',
            message=notification_results['email']['message']
        )
        log_notification(
            task_id=task['id'],
            worker_id=request.worker_id,
            notification_type='push',
            status='success' if notification_results['push']['success'] else 'failed',
            message=notification_results['push']['message']
        )
        
        print(f"📧 Email: {notification_results['email']['message']}")
        print(f"🔔 Push: {notification_results['push']['message']}")
        
        return {
            "success": True,
            "message": "Task assigned successfully!",
            "task_id": task['id'],
            "task": task,
            "notifications": notification_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign task: {str(e)}"
        )


@app.get("/worker/{worker_id}/tasks")
async def get_worker_assigned_tasks(worker_id: int, status: Optional[str] = None):
    """
    Get all tasks assigned to a worker.
    
    Args:
        worker_id: Worker ID
        status: Optional filter by status (pending, accepted, in_progress, completed, declined)
        
    Returns:
        List of tasks for the worker
    """
    try:
        tasks = get_worker_tasks(worker_id, status)
        return {
            "worker_id": worker_id,
            "total": len(tasks),
            "tasks": tasks
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get worker tasks: {str(e)}"
        )


@app.put("/task/{task_id}/status")
async def update_task_status_endpoint(task_id: int, request: TaskStatusUpdate):
    """
    Update the status of a task.
    
    Args:
        task_id: Task ID
        request: TaskStatusUpdate with new status
        
    Returns:
        Updated task details
    """
    try:
        success = update_task_status(task_id, request.status)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )
        
        task = get_task(task_id)
        return {
            "success": True,
            "message": f"Task status updated to '{request.status}'",
            "task": task
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update task status: {str(e)}"
        )


@app.get("/worker/{worker_id}/contact")
async def get_worker_contact_endpoint(worker_id: int):
    """
    Get worker contact information.
    
    Args:
        worker_id: Worker ID
        
    Returns:
        Worker contact info (email, phone, push subscription)
    """
    contact = get_worker_contact(worker_id)
    if not contact:
        return {
            "worker_id": worker_id,
            "email": None,
            "phone": None,
            "notification_enabled": True,
            "message": "No contact info on file"
        }
    return contact


@app.put("/worker/{worker_id}/contact")
async def update_worker_contact_endpoint(worker_id: int, request: WorkerContactUpdate):
    """
    Update worker contact information.
    
    Args:
        worker_id: Worker ID
        request: WorkerContactUpdate with email/phone
        
    Returns:
        Updated contact info
    """
    try:
        contact = upsert_worker_contact(
            worker_id=worker_id,
            email=request.email,
            phone=request.phone,
            notification_enabled=request.notification_enabled
        )
        return {
            "success": True,
            "message": "Contact info updated",
            "contact": contact
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update contact info: {str(e)}"
        )


@app.post("/push/subscribe")
async def subscribe_to_push(request: PushSubscription):
    """
    Subscribe a worker to push notifications.
    
    Args:
        request: PushSubscription with worker_id and subscription object
        
    Returns:
        Subscription confirmation
    """
    try:
        contact = upsert_worker_contact(
            worker_id=request.worker_id,
            push_subscription=request.subscription
        )
        return {
            "success": True,
            "message": "Push subscription saved",
            "worker_id": request.worker_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save push subscription: {str(e)}"
        )


@app.get("/tasks")
async def get_all_tasks_endpoint(status: Optional[str] = None, limit: int = 50):
    """
    Get all tasks (admin view).
    
    Args:
        status: Optional filter by status
        limit: Maximum number of tasks to return
        
    Returns:
        List of all tasks
    """
    try:
        tasks = get_all_tasks(status, limit)
        return {
            "total": len(tasks),
            "tasks": tasks
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tasks: {str(e)}"
        )


@app.get("/tasks/stats")
async def get_tasks_stats():
    """
    Get task statistics for dashboard.
    
    Returns:
        Task counts by status
    """
    try:
        stats = get_task_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Worker Recommendation API v3.0...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("💬 Chat Endpoint: http://localhost:8000/chat")
    print("📋 Task Assignment: http://localhost:8000/assign-task")
    print("📅 Appointments: http://localhost:8000/appointments")
    print("🏥 Health Check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
