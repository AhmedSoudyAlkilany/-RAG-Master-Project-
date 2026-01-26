import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User, ChevronDown, ChevronUp } from 'lucide-react';
import { queryRAG } from '../services/api';

/**
 * Chat Interface Component
 * Main chat interface for RAG interactions
 */
export default function ChatInterface({ technique = 'hybrid' }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const question = input.trim();
        setInput('');

        // Add user message
        setMessages(prev => [...prev, { role: 'user', content: question }]);
        setLoading(true);

        try {
            const response = await queryRAG(question, technique);

            // Add assistant message
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.answer,
                technique: response.technique,
                sources: response.sources,
                metadata: {
                    correctionUsed: response.correction_used,
                    evaluationResult: response.evaluation_result,
                    hybridAlpha: response.hybrid_alpha,
                }
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${error.message}`,
                isError: true
            }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="chat-interface">
            {/* Messages */}
            <div className="messages">
                {messages.length === 0 && (
                    <div className="empty-state">
                        <Bot size={48} />
                        <h3>Start a conversation</h3>
                        <p>Ask a question to query your documents using {technique.toUpperCase()} RAG</p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <Message key={i} {...msg} />
                ))}

                {loading && (
                    <div className="message assistant loading">
                        <div className="message-avatar">
                            <Bot size={20} />
                        </div>
                        <div className="message-content">
                            <Loader2 className="animate-spin" size={20} />
                            <span>Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form className="chat-input" onSubmit={handleSubmit}>
                <input
                    type="text"
                    className="input"
                    placeholder="Ask a question..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={loading}
                />
                <button
                    type="submit"
                    className="btn btn-primary btn-icon"
                    disabled={!input.trim() || loading}
                >
                    {loading ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
                </button>
            </form>

            <style>{`
        .chat-interface {
          display: flex;
          flex-direction: column;
          height: 100%;
        }
        
        .messages {
          flex: 1;
          overflow-y: auto;
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }
        
        .empty-state {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          color: var(--text-muted);
          gap: 1rem;
        }
        
        .empty-state h3 {
          color: var(--text-secondary);
          font-weight: 500;
        }
        
        .message {
          display: flex;
          gap: 1rem;
          animation: fadeIn 0.3s ease-out;
        }
        
        .message.user {
          flex-direction: row-reverse;
        }
        
        .message-avatar {
          width: 36px;
          height: 36px;
          border-radius: var(--radius-md);
          background: var(--bg-tertiary);
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }
        
        .message.user .message-avatar {
          background: var(--accent-gradient);
          color: white;
        }
        
        .message-content {
          max-width: 70%;
          background: var(--bg-tertiary);
          padding: 1rem 1.25rem;
          border-radius: var(--radius-lg);
        }
        
        .message.user .message-content {
          background: var(--accent-primary);
          color: white;
        }
        
        .message.loading .message-content {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: var(--text-muted);
        }
        
        .message.error .message-content {
          background: rgba(239, 68, 68, 0.2);
          border: 1px solid var(--error);
        }
        
        .chat-input {
          display: flex;
          gap: 0.75rem;
          padding: 1rem 1.5rem;
          border-top: 1px solid var(--glass-border);
          background: var(--bg-secondary);
        }
        
        .chat-input .input {
          flex: 1;
        }
        
        /* Sources */
        .sources-toggle {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-top: 0.75rem;
          padding-top: 0.75rem;
          border-top: 1px solid var(--glass-border);
          color: var(--text-muted);
          font-size: 0.85rem;
          cursor: pointer;
          background: none;
          border: none;
          width: 100%;
          text-align: left;
        }
        
        .sources-toggle:hover {
          color: var(--text-secondary);
        }
        
        .sources-list {
          margin-top: 0.75rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        
        .source-item {
          background: rgba(0,0,0,0.2);
          padding: 0.75rem;
          border-radius: var(--radius-sm);
          font-size: 0.85rem;
        }
        
        .source-item .source-label {
          font-size: 0.75rem;
          color: var(--text-muted);
          margin-bottom: 0.25rem;
        }
        
        .technique-badge {
          display: inline-block;
          padding: 0.125rem 0.5rem;
          font-size: 0.7rem;
          font-weight: 600;
          text-transform: uppercase;
          border-radius: 4px;
          margin-top: 0.5rem;
        }
        
        .technique-badge.hybrid { background: rgba(6, 182, 212, 0.2); color: var(--hybrid-color); }
        .technique-badge.rerank { background: rgba(249, 115, 22, 0.2); color: var(--rerank-color); }
        .technique-badge.crag { background: rgba(139, 92, 246, 0.2); color: var(--crag-color); }
        .technique-badge.naive { background: rgba(100, 116, 139, 0.2); color: var(--naive-color); }
      `}</style>
        </div>
    );
}


// Message Component
function Message({ role, content, technique, sources, metadata, isError }) {
    const [showSources, setShowSources] = useState(false);

    return (
        <div className={`message ${role} ${isError ? 'error' : ''}`}>
            <div className="message-avatar">
                {role === 'user' ? <User size={20} /> : <Bot size={20} />}
            </div>
            <div className="message-content">
                <div style={{ whiteSpace: 'pre-wrap' }}>{content}</div>

                {technique && (
                    <span className={`technique-badge ${technique}`}>{technique}</span>
                )}

                {metadata?.correctionUsed && (
                    <span className="technique-badge crag" style={{ marginLeft: '0.5rem' }}>
                        Web Search Used
                    </span>
                )}

                {sources && sources.length > 0 && (
                    <>
                        <button
                            className="sources-toggle"
                            onClick={() => setShowSources(!showSources)}
                        >
                            {showSources ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            {sources.length} source{sources.length !== 1 ? 's' : ''}
                        </button>

                        {showSources && (
                            <div className="sources-list">
                                {sources.map((source, i) => (
                                    <div key={i} className="source-item">
                                        <div className="source-label">
                                            {source.type === 'web' ? '🌐 Web' : '📄 Local'} • {source.metadata?.filename || source.metadata?.source || 'Unknown'}
                                        </div>
                                        <div>{source.content?.slice(0, 200)}...</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
