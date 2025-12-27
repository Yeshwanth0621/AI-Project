import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const ChatInterface = () => {
    const [messages, setMessages] = useState([
        { role: 'system', content: 'System initialized. Ready for input.' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        // Placeholder for AI response
        setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

        try {
            const response = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage.content }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                setMessages(prev => {
                    const newMessages = [...prev];
                    const lastMessage = newMessages[newMessages.length - 1];
                    lastMessage.content += chunk;
                    return newMessages;
                });
            }
        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, { role: 'system', content: 'Error: Connection lost.' }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="w-full max-w-4xl h-[90vh] flex flex-col glass-panel overflow-hidden relative">
            {/* Header */}
            <div className="p-4 border-b border-[rgba(255,255,255,0.1)] flex items-center justify-between bg-[rgba(0,0,0,0.2)]">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-full bg-[rgba(0,243,255,0.1)] border border-[rgba(0,243,255,0.3)]">
                        <Sparkles size={20} className="text-[var(--primary-color)]" />
                    </div>
                    <h1 className="text-xl font-bold tracking-wider text-white">NEXUS <span className="text-[var(--primary-color)]">AI</span></h1>
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-400">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    ONLINE
                </div>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                <AnimatePresence>
                    {messages.map((msg, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.3 }}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div className={`flex gap-4 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>

                                {/* Avatar */}
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 
                  ${msg.role === 'user'
                                        ? 'bg-[rgba(188,19,254,0.1)] border border-[rgba(188,19,254,0.3)]'
                                        : msg.role === 'system'
                                            ? 'bg-gray-800 border border-gray-700'
                                            : 'bg-[rgba(0,243,255,0.1)] border border-[rgba(0,243,255,0.3)]'
                                    }`}
                                >
                                    {msg.role === 'user' ? <User size={18} className="text-[var(--secondary-color)]" /> :
                                        msg.role === 'system' ? <Bot size={18} className="text-gray-400" /> :
                                            <Bot size={18} className="text-[var(--primary-color)]" />}
                                </div>

                                {/* Bubble */}
                                <div className={`p-4 rounded-2xl backdrop-blur-sm border 
                  ${msg.role === 'user'
                                        ? 'bg-[rgba(188,19,254,0.05)] border-[rgba(188,19,254,0.2)] text-white rounded-tr-none'
                                        : msg.role === 'system'
                                            ? 'bg-transparent border-transparent text-gray-500 text-sm italic'
                                            : 'bg-[rgba(0,243,255,0.05)] border-[rgba(0,243,255,0.2)] text-gray-100 rounded-tl-none'
                                    }`}
                                >
                                    <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 bg-[rgba(0,0,0,0.3)] border-t border-[rgba(255,255,255,0.1)]">
                <form onSubmit={handleSubmit} className="relative flex items-center gap-4">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Enter command..."
                        className="w-full bg-[rgba(255,255,255,0.03)] border border-[rgba(255,255,255,0.1)] rounded-xl px-6 py-4 text-white placeholder-gray-500 focus:outline-none focus:border-[var(--primary-color)] focus:shadow-[0_0_15px_rgba(0,243,255,0.1)] transition-all duration-300"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="absolute right-2 p-2 rounded-lg bg-[var(--primary-color)] text-black hover:bg-cyan-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        <Send size={20} />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ChatInterface;
