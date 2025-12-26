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
  const [typingResponse, setTypingResponse] = useState(''); // Current typing text
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [welcomeTyping, setWelcomeTyping] = useState(''); // Welcome typing text
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
  const welcomeIntervalRef = useRef(null);
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'https://your-backend.onrender.com';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => scrollToBottom(), [messages, typingResponse, welcomeTyping]);

  useEffect(() => {
    if (messages.length === 0 && welcomeTyping === '') {
      const welcomeMessage = "Hey there! I'm Nova, your witty AI sidekick built by Sanu Sharma. What's sparking your curiosity today? Drop a question or snap an image to dive in!";
      let index = 0;
      welcomeIntervalRef.current = setInterval(() => {
        if (index < welcomeMessage.length) {
          setWelcomeTyping(prev => prev + welcomeMessage.charAt(index));
          index++;
        } else {
          clearInterval(welcomeIntervalRef.current);
        }
      }, 40); // Typing speed for welcome
      return () => clearInterval(welcomeIntervalRef.current);
    }
  }, [messages.length, welcomeTyping]);

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
    }, 20); // Slightly faster for modern feel
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

  const toggleDarkMode = () => {
    setIsDarkMode(prev => !prev);
  };

  return (
    <div className={`app ${isDarkMode ? 'dark' : ''}`}>
      <header className="header">
        <div className="header-content">
          <div className="header-avatar">N</div>
          <h1 className="header-title">Nova AI Chat</h1>
          <div className="header-status">Online</div>
          <button onClick={toggleDarkMode} className="dark-toggle" title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
            {isDarkMode ? '☀️' : '🌙'}
          </button>
        </div>
      </header>
      <main className="main">
        <div className="chat-messages">
          {messages.length === 0 && welcomeTyping !== '' && (
            <div className="welcome-message">
              <div className="welcome-avatar">N</div>
              <div className="welcome-content">
                <p className="welcome-text" dangerouslySetInnerHTML={{ __html: welcomeTyping.replace(/\n/g, '<br/>') }} />
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message-wrapper ${msg.role} slide-in-${msg.role}`}>
              <div className={`message ${msg.role}`}>
                <div className={`avatar ${msg.role}`}>{msg.role === 'user' ? 'U' : 'N'}</div>
                <div className="message-content">
                  {msg.image && <img src={msg.image} alt="Uploaded" className="uploaded-image" />}
                  <p dangerouslySetInnerHTML={{ __html: typeof msg.content === 'string' ? msg.content.replace(/\n/g, '<br/>') : 'Image query' }} />
                </div>
              </div>
            </div>
          ))}
          {(loading || typingResponse) && (
            <div className="message-wrapper assistant slide-in-assistant">
              <div className="message assistant">
                <div className="avatar assistant">N</div>
                <div className="message-content typing-message">
                  <div className="typing-indicator">
                    <span className="dot"></span>
                    <span className="dot"></span>
                    <span className="dot"></span>
                  </div>
                  {typingResponse && (
                    <p className="typing-text" dangerouslySetInnerHTML={{ __html: typingResponse.replace(/\n/g, '<br/>') }} />
                  )}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>
      <div className="input-area">
        <div className="input-wrapper">
          <label className="file-input">
            <input type="file" accept="image/*" onChange={handleImageUpload} disabled={loading} />
            <span className="file-icon">📷</span>
          </label>
          {imagePreview && <img src={imagePreview} alt="Preview" className="preview-image" />}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
            placeholder="Type your message... (Shift+Enter for new line)"
            disabled={loading || typingResponse}
            className="input-field"
            rows={1}
          />
          <button onClick={sendMessage} disabled={loading || typingResponse || (!input.trim() && !image)} className="send-btn">
            <span className="send-icon">→</span>
          </button>
        </div>
      </div>
      <footer className="footer">
        <p>Built with ❤️ by <strong>Sanu Sharma</strong></p>
      </footer>
    </div>
  );
}

export default App;