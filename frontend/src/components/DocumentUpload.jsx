import { useState, useRef, useEffect } from 'react';
import { Upload, FileText, Trash2, Loader2, CheckCircle, AlertCircle, File } from 'lucide-react';
import { uploadDocument, listDocuments, clearDocuments } from '../services/api';

/**
 * Document Upload Component
 */
export default function DocumentUpload({ compact = false }) {
    const [documents, setDocuments] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState(null);
    const fileInputRef = useRef(null);

    // Load documents on mount
    useEffect(() => {
        loadDocuments();
    }, []);

    const loadDocuments = async () => {
        try {
            const result = await listDocuments();
            setDocuments(result.documents || []);
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    };

    const handleUpload = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;

        setUploading(true);
        setUploadStatus(null);

        try {
            for (const file of files) {
                const result = await uploadDocument(file);
                setUploadStatus({ success: true, message: `Uploaded ${file.name} (${result.num_chunks} chunks)` });
            }
            await loadDocuments();
        } catch (error) {
            setUploadStatus({ success: false, message: error.message });
        } finally {
            setUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const handleClear = async () => {
        if (!confirm('Are you sure you want to delete all documents?')) return;

        try {
            await clearDocuments();
            setDocuments([]);
            setUploadStatus({ success: true, message: 'All documents cleared' });
        } catch (error) {
            setUploadStatus({ success: false, message: error.message });
        }
    };

    if (compact) {
        return (
            <div className="upload-compact">
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleUpload}
                    accept=".pdf,.txt,.md,.docx,.doc"
                    multiple
                    style={{ display: 'none' }}
                />
                <button
                    className="btn btn-secondary w-full"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading}
                >
                    {uploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                    Upload File
                </button>
                {documents.length > 0 && (
                    <div className="doc-count">{documents.length} file(s) loaded</div>
                )}

                <style>{`
          .upload-compact {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
          }
          .doc-count {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
          }
        `}</style>
            </div>
        );
    }

    return (
        <div className="document-upload">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleUpload}
                accept=".pdf,.txt,.md,.docx,.doc"
                multiple
                style={{ display: 'none' }}
            />

            {/* Drop Zone */}
            <div
                className="drop-zone"
                onClick={() => fileInputRef.current?.click()}
            >
                <Upload size={48} />
                <h3>Drop files here or click to upload</h3>
                <p>Supports PDF, TXT, MD, DOCX, DOC</p>
                {uploading && (
                    <div className="uploading">
                        <Loader2 className="animate-spin" size={24} />
                        <span>Uploading...</span>
                    </div>
                )}
            </div>

            {/* Status */}
            {uploadStatus && (
                <div className={`status-message ${uploadStatus.success ? 'success' : 'error'}`}>
                    {uploadStatus.success ? <CheckCircle size={18} /> : <AlertCircle size={18} />}
                    {uploadStatus.message}
                </div>
            )}

            {/* Document List */}
            <div className="document-list">
                <div className="doc-header">
                    <h3>Uploaded Documents ({documents.length})</h3>
                    {documents.length > 0 && (
                        <button className="btn btn-secondary" onClick={handleClear}>
                            <Trash2 size={16} /> Clear All
                        </button>
                    )}
                </div>

                {documents.length === 0 ? (
                    <div className="empty">No documents uploaded yet</div>
                ) : (
                    <div className="doc-grid">
                        {documents.map((doc, i) => (
                            <div key={i} className="doc-item">
                                <FileText size={24} />
                                <div className="doc-info">
                                    <div className="doc-name">{doc.filename}</div>
                                    <div className="doc-chunks">{doc.chunks} chunks</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <style>{`
        .document-upload {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          margin-top: 1.5rem;
        }
        
        .drop-zone {
          border: 2px dashed var(--glass-border);
          border-radius: var(--radius-lg);
          padding: 3rem;
          text-align: center;
          cursor: pointer;
          transition: all var(--transition-normal);
          background: var(--bg-card);
        }
        
        .drop-zone:hover {
          border-color: var(--accent-primary);
          background: rgba(99, 102, 241, 0.1);
        }
        
        .drop-zone h3 {
          margin: 1rem 0 0.5rem;
        }
        
        .drop-zone p {
          color: var(--text-muted);
        }
        
        .uploading {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.75rem;
          margin-top: 1rem;
          color: var(--accent-primary);
        }
        
        .status-message {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem;
          border-radius: var(--radius-md);
        }
        
        .status-message.success {
          background: rgba(34, 197, 94, 0.1);
          color: var(--success);
          border: 1px solid rgba(34, 197, 94, 0.3);
        }
        
        .status-message.error {
          background: rgba(239, 68, 68, 0.1);
          color: var(--error);
          border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .document-list {
          background: var(--bg-card);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          padding: 1.5rem;
        }
        
        .doc-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        
        .doc-header h3 {
          font-size: 1rem;
        }
        
        .empty {
          color: var(--text-muted);
          text-align: center;
          padding: 2rem;
        }
        
        .doc-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 1rem;
        }
        
        .doc-item {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 1rem;
          background: var(--bg-tertiary);
          border-radius: var(--radius-md);
        }
        
        .doc-info {
          flex: 1;
          min-width: 0;
        }
        
        .doc-name {
          font-weight: 500;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        
        .doc-chunks {
          font-size: 0.8rem;
          color: var(--text-muted);
        }
      `}</style>
        </div>
    );
}
