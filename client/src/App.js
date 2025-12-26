import React, { useState, useRef, useEffect, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import './Chat.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [typingResponse, setTypingResponse] = useState(''); // Current typing text
  const [showCursor, setShowCursor] = useState(true); // For blinking cursor effect
  const typingBufferRef = useRef(''); // Ref for buffer
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
  const welcomeIntervalRef = useRef(null);
  const typingTimeoutRef = useRef(null);
  const cursorIntervalRef = useRef(null); // For cursor blink
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

  // Blinking cursor effect
  useEffect(() => {
    if (loading || typingResponse) {
      cursorIntervalRef.current = setInterval(() => {
        setShowCursor(prev => !prev);
      }, 500); // Blink every 500ms
    } else {
      setShowCursor(false);
    }
    return () => {
      if (cursorIntervalRef.current) {
        clearInterval(cursorIntervalRef.current);
      }
    };
  }, [loading, typingResponse]);

  const startTyping = useCallback(() => {
    if (typingTimeoutRef.current) return;

    const typeNext = () => {
      if (typingBufferRef.current === '') {
        if (typingTimeoutRef.current) {
          clearTimeout(typingTimeoutRef.current);
          typingTimeoutRef.current = null;
        }
        return;
      }
      const nextChar = typingBufferRef.current.charAt(0);
      setTypingResponse(prev => prev + nextChar);
      typingBufferRef.current = typingBufferRef.current.slice(1);

      let delay = 50; // Base delay
      if (nextChar === ' ' || /[,.;!?]/.test(nextChar)) {
        delay = 100; // Pause after punctuation/spaces
      } else if (/[a-zA-Z]/.test(nextChar)) {
        delay = 40; // Faster for letters
      }

      typingTimeoutRef.current = setTimeout(typeNext, delay);
    };

    typeNext();
  }, []);

  const clearTyping = useCallback(() => {
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
      typingTimeoutRef.current = null;
    }
    if (cursorIntervalRef.current) {
      clearInterval(cursorIntervalRef.current);
      cursorIntervalRef.current = null;
    }
    typingBufferRef.current = '';
    setTypingResponse('');
    setShowCursor(false);
  }, []);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const flushAndClearTyping = useCallback(() => {
    // Flush remaining buffer instantly into typingResponse
    if (typingBufferRef.current) {
      setTypingResponse(prev => prev + typingBufferRef.current);
      typingBufferRef.current = '';
    }
    // Clear timeout if running
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
      typingTimeoutRef.current = null;
    }
    setTypingResponse(''); // Hide placeholder after flush
    setShowCursor(false);
  }, []);

  const sendMessage = () => {
    if (!input.trim() && !imagePreview) return;

    // Reset typing states before new request
    clearTyping();
    setLoading(true);

    if (imagePreview) {
      // For images, use POST endpoint with fetch and manual SSE parsing
      const userContent = input.trim() || "Describe this image.";
      let payloadContent = [
        { type: "text", text: userContent },
        { type: "image_url", image_url: { url: imagePreview } }
      ];

      setMessages(prev => [...prev, { role: 'user', content: userContent, image: imagePreview }]);
      setInput('');

      fetch(`${backendUrl}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          messages: [{ role: "user", content: payloadContent }]
        })
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let finalText = '';

          function parseEvent(eventStr) {
            const lines = eventStr.split('\n');
            let eventType = 'message';
            let data = '';
            for (let line of lines) {
              if (line.startsWith('event: ')) {
                eventType = line.slice(7).trim();
              } else if (line.startsWith('data: ')) {
                data += line.slice(6);
              }
            }
            // Remove newlines but preserve spaces
            data = data.replace(/\n/g, '');
            if (!data) return false;

            if (eventType === 'end') {
              try {
                const endData = JSON.parse(data);
                const usedModel = endData.model;
                const usedModelNote = usedModel ? ` (via ${usedModel})` : '';
                flushAndClearTyping();
                setMessages(prev => [...prev, { role: 'assistant', content: finalText + usedModelNote }]);
              } catch (e) {
                flushAndClearTyping();
                setMessages(prev => [...prev, { role: 'assistant', content: finalText }]);
              }
              setLoading(false);
              setImage(null);
              setImagePreview(null);
              reader.cancel();
              return true; // End of stream
            } else if (eventType === 'error') {
              clearTyping();
              try {
                const errData = JSON.parse(data);
                setTypingResponse(`Error: ${errData.error}`);
              } catch (e) {
                setTypingResponse(`Error: ${data}`);
              }
              setLoading(false);
              setImage(null);
              setImagePreview(null);
              reader.cancel();
              return true;
            } else {
              // Default message data (tokens)
              finalText += data;
              typingBufferRef.current += data;
              startTyping();
            }
            return false;
          }

          let buffer = '';

          function read() {
            reader.read().then(({ done, value }) => {
              if (done) {
                if (typingBufferRef.current) {
                  // Flush any remaining on end of stream
                  flushAndClearTyping();
                }
                return;
              }
              buffer += decoder.decode(value, { stream: true });
              let boundary;
              while ((boundary = buffer.indexOf('\n\n')) !== -1) {
                const event = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);
                if (parseEvent(event)) {
                  return; // Stream ended
                }
              }
              read();
            }).catch(err => {
              console.error('Stream error:', err);
              clearTyping();
              setTypingResponse("⚠️ Connection lost.");
              setLoading(false);
              setImage(null);
              setImagePreview(null);
            });
          }
          read();
        })
        .catch(err => {
          console.error('Fetch error:', err);
          clearTyping();
          setTypingResponse(`Error: ${err.message}`);
          setLoading(false);
          setImage(null);
          setImagePreview(null);
        });
    } else {
      // For text-only, use GET EventSource
      const userContent = input.trim();
      const url = new URL(`${backendUrl}/api/chat/stream`);
      url.searchParams.set("session_id", sessionId);
      url.searchParams.set("message", userContent);

      setMessages(prev => [...prev, { role: 'user', content: userContent }]);
      setInput('');

      const es = new EventSource(url.toString());

      let finalText = '';

      es.onmessage = (e) => {
        // Remove newlines but preserve spaces; no trim
        const data = e.data.replace(/\n/g, '');
        if (data) {
          finalText += data;
          typingBufferRef.current += data;
          startTyping();
        }
      };

      es.addEventListener("end", (e) => {
        // For end event, data is JSON, remove \n if any
        const endDataStr = e.data.replace(/\n/g, '');
        flushAndClearTyping();
        try {
          const endData = JSON.parse(endDataStr);
          const usedModel = endData.model;
          const usedModelNote = usedModel ? ` (via ${usedModel})` : '';
          setMessages(prev => [...prev, { role: 'assistant', content: finalText + usedModelNote }]);
        } catch {
          setMessages(prev => [...prev, { role: 'assistant', content: finalText }]);
        }
        setLoading(false);
        es.close();
      });

      es.addEventListener("error", (err) => {
        console.error('EventSource failed:', err);
        clearTyping();
        setTypingResponse("⚠️ Connection lost.");
        setLoading(false);
        es.close();
      });

      // Cleanup on unmount or close
      return () => {
        es.close();
      };
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
                  {loading && !typingResponse && (
                    <div className="typing-indicator">
                      <span className="dot"></span>
                      <span className="dot"></span>
                      <span className="dot"></span>
                    </div>
                  )}
                  {typingResponse && (
                    <>
                      <p className="typing-text" dangerouslySetInnerHTML={{ __html: typingResponse.replace(/\n/g, '<br/>') }} />
                      {showCursor && <span className="blinking-cursor">|</span>}
                    </>
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