import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import './Chat.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [typingResponse, setTypingResponse] = useState('');  // Current typing text
  const sessionId = useState(() => {
    let id = localStorage.getItem('chatSessionId');
    if (!id) {
      id = uuidv4();
      localStorage.setItem('chatSessionId', id);
    }
    return id;
  })[0];
  const messagesEndRef = useRef(null);
  const intervalRef = useRef(null);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'https://your-backend.onrender.com';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => scrollToBottom(), [messages, typingResponse]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const startTypingEffect = (text) => {
    setTypingResponse('');
    let index = 0;
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      if (index < text.length) {
        setTypingResponse(prev => prev + text.charAt(index));
        index++;
      } else {
        clearInterval(intervalRef.current);
        // Typing complete → add permanent message
        setMessages(prev => [...prev, { role: 'assistant', content: text }]);
        setTypingResponse('');
      }
    }, 50);  // Cool typing speed
  };

  const sendMessage = async () => {
    if (!input.trim() && !image) return;

    const userContent = input.trim() || "Describe this image.";
    let userMsgContent = userContent;
    if (imagePreview) {
      userMsgContent = [
        { type: "text", text: userContent },
        { type: "image_url", image_url: { url: imagePreview } }
      ];
    }

    const userMessage = { role: 'user', content: userMsgContent, image: imagePreview };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post(`${backendUrl}/api/chat`, {
        messages: [{ content: userMsgContent }],
        session_id: sessionId
      });
      const usedModelNote = res.data.used_model ? ` (via ${res.data.used_model})` : '';
      const fullText = res.data.response + usedModelNote;
      startTypingEffect(fullText);
    } catch (err) {
      const errorText = `Error: ${err.response?.data?.detail || err.message}`;
      startTypingEffect(errorText);
    } finally {
      setLoading(false);
      setImage(null);
      setImagePreview(null);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">Groq AI Chat</h1>
        <p className="subtitle">Seamless & Smart Conversations</p>
      </header>

      <div className="chat-container">
        <div className="chat-window">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role} fade-in`}>
              {msg.image && <img src={msg.image} alt="Uploaded" className="uploaded-image" />}
              <div className="message-content">
                <p dangerouslySetInnerHTML={{ __html: typeof msg.content === 'string' ? msg.content.replace(/\n/g, '<br/>') : 'Image query' }} />
              </div>
            </div>
          ))}
          {(loading || typingResponse) && (
            <div className="message assistant fade-in">
              <div className="message-content typing-text">
                <p dangerouslySetInnerHTML={{ __html: typingResponse.replace(/\n/g, '<br/>') + '<span class="cursor">|</span>' }} />
              </div>
              {loading && (
                <div className="typing-indicator">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area slide-up">
          <label className="file-input">
            <input type="file" accept="image/*" onChange={handleImageUpload} />
            <span>📷 Choose Image (optional)</span>
          </label>
          {imagePreview && <img src={imagePreview} alt="Preview" className="preview-image" />}
          <div className="text-input-wrapper">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Type your message..."
              disabled={loading || typingResponse}
              className="input-field"
            />
            <button onClick={sendMessage} disabled={loading || typingResponse || (!input.trim() && !image)} className="send-btn">
              Send 🚀
            </button>
          </div>
        </div>
      </div>

      <footer className="footer">
        <p>Built by <strong>Sanu Sharma</strong> 🎄</p>
      </footer>
    </div>
  );
}

export default App;