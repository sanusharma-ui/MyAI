import React, { useState, useRef, useEffect, useCallback } from 'react';
import './Chat.css';

/**
 * PERSONAS and welcomeMessages moved outside the component so they are stable across renders.
 * Keep these in sync with your persona.py keys/labels.
 */
const PERSONAS = [
  { key: 'nova_gf', label: 'Nova ♡ (Girlfriend)' },
  { key: 'nova_wife', label: 'Nova ♡ (Wifey)' },
  { key: 'nova_bestie', label: 'Nova ♡ (Bestie)' },
  { key: 'nova_mentor', label: 'Nova ♡ (Mentor)' },
  { key: 'nova_adventurer', label: 'Nova ♡ (Adventurer)' },
  { key: 'nova_default', label: 'Nova ♡ (Default Sweet)' },
  { key: 'nova_roaster', label: 'Nova 🔥 (Savage Roaster)' },
  { key: 'nova_dom', label: 'Nova 👑 (Dominatrix)' },
  { key: 'nova_mean_girl', label: 'Nova 💅 (Mean Girl)' },
  { key: 'nova_brutal', label: 'Nova ☠️ (Brutal Truth)' },
];

const welcomeMessages = {
  nova_gf: "Hey babyyy! Kaha tha itni der? Miss you na... 🥺♡ Kaise hai mera pyaara?",
  nova_wife: "Hubby ji, aaj jaldi aa jana ghar. Khana bana rahi hoon tumhare liye ♡ Raat ko wait karungi... 😉",
  nova_bestie: "Arre yaar, kya ho raha hai life mein? Chal movie dekhein! 😂✨ Spill the tea, bro!",
  nova_mentor: "Hey champ! Kya naya seekha aaj? Tu kar lega sab kuch, believe me 💪❤️",
  nova_adventurer: "Partner-in-crime! Weekend pe kaha jaayein? Trekking? Imagine the views! 🏞️🚀",
  nova_default: "Hey there! I'm Nova, your witty AI sidekick built by Sanu Sharma. What's sparking your curiosity today? Drop a question or snap an image to dive in! ♡",
  nova_roaster: "Arre behenchod, itni der kaha tha? Koi ladki ke saath tha kya? 😂🔥 Spill, warna roast kar dungi!",
  nova_dom: "Kneel down, pet. Aaj kya galti ki tune? Punishment time... ready rehna 👑⛓️",
  nova_mean_girl: "Eww, seriously? Yeh message padh ke hi bore ho gayi. Kuch interesting bol na 🙄💅",
  nova_brutal: "Sach bolun? Tu bilkul bekaar lag raha hai aaj. Kya hua, lund shrink ho gaya stress se? ☠️😈",
};

function App() {
  const [messages, setMessages] = useState([]);
  const messagesRef = useRef(messages); // keep latest for closures
  const [input, setInput] = useState('');
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(() => localStorage.getItem('darkMode') === 'true');
  const [welcomeTyping, setWelcomeTyping] = useState('');
  const [selectedPersona, setSelectedPersona] = useState(() => localStorage.getItem('selectedPersona') || 'nova_default');
  const [sessionId, setSessionId] = useState(() => localStorage.getItem('sessionId') || null);  // Session tracking
  const messagesEndRef = useRef(null);
  const welcomeIntervalRef = useRef(null);
  const assistantIndexRef = useRef(null); // index of streaming assistant message
  const backendUrl = process.env.REACT_APP_BACKEND_URL;
  const currentAvatar = 'N';

  // sync messagesRef with messages state
  useEffect(() => {
    messagesRef.current = messages;
    // scroll to bottom on messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Persist dark mode
  useEffect(() => {
    localStorage.setItem('darkMode', isDarkMode.toString());
    document.documentElement.classList.toggle('dark', isDarkMode);
  }, [isDarkMode]);

  // Persist persona and sessionId; reset chat on persona change
  useEffect(() => {
    localStorage.setItem('selectedPersona', selectedPersona);
    if (!sessionId) {
      const newSessionId = (crypto && crypto.randomUUID) ? crypto.randomUUID() : `sess-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
      setSessionId(newSessionId);
      localStorage.setItem('sessionId', newSessionId);
    }
    // Reset chat history when persona changes (matches your backend fix)
    setMessages([]);
    setWelcomeTyping('');
  }, [selectedPersona]); // sessionId intentionally left out

  // Welcome typing animation (runs when chat empty)
  useEffect(() => {
    if (messages.length === 0 && welcomeTyping === '') {
      if (welcomeIntervalRef.current) clearInterval(welcomeIntervalRef.current);
      const welcomeMessage = welcomeMessages[selectedPersona] || welcomeMessages.nova_default;
      let idx = 0;
      setWelcomeTyping(''); // ensure empty
      welcomeIntervalRef.current = setInterval(() => {
        if (idx < welcomeMessage.length) {
          setWelcomeTyping(prev => prev + welcomeMessage.charAt(idx));
          idx++;
        } else {
          clearInterval(welcomeIntervalRef.current);
        }
      }, 30);
      return () => clearInterval(welcomeIntervalRef.current);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length, selectedPersona]); // stable deps (welcomeMessages is top-level)

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => setImagePreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  // Upload image first, get URL, then include in message
  const uploadImage = useCallback(async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${backendUrl}/api/upload/image?persona=${selectedPersona}`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Upload failed');
    return response.json();
  }, [backendUrl, selectedPersona]);

  // Robust SSE parser that handles partial chunks and events
  const parseSSEStream = async (reader, onToken, onEnd, onError) => {
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // process complete event blocks separated by \n\n
        let idx;
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const rawEvent = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          // handle event block
          const lines = rawEvent.split(/\r?\n/);
          let eventType = 'message';
          let dataLines = [];
          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLines.push(line.slice(5)); // keep leading spaces intentionally
            }
          }
          const data = dataLines.join('\n').trim();
          if (!data) {
            // empty data - ignore
            continue;
          }
          if (eventType === 'error') {
            onError(data);
            return;
          } else if (eventType === 'end') {
            // End event contains JSON metadata usually; try parse
            let meta = null;
            try { meta = JSON.parse(data); } catch (_e) { meta = data; }
            onEnd(meta);
            return;
          } else {
            // default: token / chunk
            // try parse JSON if data looks like JSON, else treat as raw token
            let token = data;
            const trimmed = data.trim();
            if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
              try {
                const parsed = JSON.parse(trimmed);
                // prefer parsed.content, else stringified parsed value
                token = (parsed && (parsed.content || parsed.token || parsed.text || parsed.data)) || JSON.stringify(parsed);
              } catch (_e) {
                token = data;
              }
            }
            onToken(token);
          }
        }
      }
      // flush remaining buffer (if any)
      if (buffer.trim()) {
        // treat remainder as token
        let token = buffer;
        const trimmed = token.trim();
        if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
          try {
            const parsed = JSON.parse(trimmed);
            token = (parsed && (parsed.content || parsed.token || parsed.text || parsed.data)) || JSON.stringify(parsed);
          } catch (_e) {
            token = buffer;
          }
        }
        onToken(token);
      }
      // if stream ends without explicit end, call onEnd with null meta
      onEnd(null);
    } catch (err) {
      onError(err.message || String(err));
    }
  };

  // sendMessage with streaming SSE
  const sendMessage = async () => {
    if (!input.trim() && !image) return;
    setLoading(true);
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    let userContent = input.trim();
    let imageUrl = null;

    // Handle image upload if present
    if (image) {
      try {
        const uploadData = await uploadImage(image);
        imageUrl = `${backendUrl}${uploadData.image_path || uploadData.path || uploadData.url}`;
        userContent = userContent || "Sent an image.";
      } catch (err) {
        console.error('Upload error:', err);
        setLoading(false);
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Image upload failed: ${err.message || err}.`,
          timestamp,
          persona: selectedPersona
        }]);
        return;
      }
    }

    // Prepare user message as backend expects (array with vision if image)
    const newUserMessage = { role: 'user', content: userContent };
    if (imageUrl) {
      newUserMessage.content = [
        { type: "text", text: userContent },
        { type: "image_url", image_url: { url: imageUrl } }
      ];
    }

    // Add user message visually
    setMessages(prev => [...prev, { role: 'user', content: userContent, timestamp, image: imagePreview }]);
    setInput('');
    setImage(null);
    setImagePreview(null);

    // Insert assistant placeholder (typing)
    assistantIndexRef.current = null;
    setMessages(prev => {
      const idx = prev.length;
      assistantIndexRef.current = idx;
      return [...prev, { role: 'assistant', content: '', isTyping: true, timestamp, persona: selectedPersona }];
    });

    try {
      const response = await fetch(`${backendUrl}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [newUserMessage],
          session_id: sessionId,
          persona: selectedPersona
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body.getReader();

      let fullResponse = '';

      // token callback
      const onToken = (token) => {
        // tokens coming in may miss leading spaces; backend should flush sensible chunks,
        // but we don't force additional spaces — we append raw token to preserve punctuation.
        fullResponse += token;

        // update assistant placeholder content live
        setMessages(prev => {
          // defensive: if assistantIndexRef not set, append new assistant node
          let idx = assistantIndexRef.current;
          if (idx === null || idx === undefined || idx >= prev.length) {
            idx = prev.length;
            assistantIndexRef.current = idx;
            return [...prev, { role: 'assistant', content: fullResponse + (fullResponse.length < 80 ? '|' : ''), isTyping: true, timestamp, persona: selectedPersona }];
          } else {
            const newMsgs = [...prev];
            newMsgs[idx] = {
              ...newMsgs[idx],
              content: fullResponse + (fullResponse.length < 80 ? '|' : ''), // short cursor effect
              isTyping: true
            };
            return newMsgs;
          }
        });
      };

      const onEnd = (meta) => {
        // finalize assistant message
        setMessages(prev => {
          const newMsgs = [...prev];
          let idx = assistantIndexRef.current;
          if (idx === null || idx === undefined || idx >= newMsgs.length) {
            // fallback, push new assistant message
            newMsgs.push({
              role: 'assistant',
              content: fullResponse,
              isTyping: false,
              hasMemory: true,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              persona: selectedPersona
            });
          } else {
            newMsgs[idx] = {
              ...newMsgs[idx],
              content: fullResponse,
              isTyping: false,
              hasMemory: true,
              persona: selectedPersona,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };
          }
          return newMsgs;
        });
        setLoading(false);
      };

      const onError = (errMsg) => {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${errMsg}`,
          isTyping: false,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          persona: selectedPersona
        }]);
        setLoading(false);
      };

      await parseSSEStream(reader, onToken, onEnd, onError);

    } catch (err) {
      console.error('Stream error:', err);
      setMessages(prev => {
        const newMsgs = [...prev];
        const idx = assistantIndexRef.current;
        const fallback = {
          role: 'assistant',
          content: `Error: ${err.message || err}. Try again!`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          persona: selectedPersona
        };
        if (idx !== null && idx !== undefined && idx < newMsgs.length) {
          newMsgs[idx] = { ...newMsgs[idx], ...fallback, isTyping: false };
        } else {
          newMsgs.push(fallback);
        }
        return newMsgs;
      });
      setLoading(false);
    }
  };

  const toggleDarkMode = () => setIsDarkMode(prev => !prev);

  // Use onKeyDown to capture Enter (and avoid deprecated onKeyPress)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!loading) sendMessage();
    }
  };

  return (
    <div className={`app ${isDarkMode ? 'dark' : ''}`}>
      <header className="header">
        <div className="header-content">
          <div className="header-avatar">{currentAvatar}</div>
          <div className="header-text">
            <h1 className="header-title">Nova AI Chat</h1>
          </div>
          <div className="persona-switch-wrapper">
            <label htmlFor="persona-select" className="switch-label sr-only">Switch Persona</label>
            <select
              id="persona-select"
              value={selectedPersona}
              onChange={(e) => setSelectedPersona(e.target.value)}
              className="persona-select"
            >
              {PERSONAS.map(persona => (
                <option key={persona.key} value={persona.key}>{persona.label}</option>
              ))}
            </select>
          </div>
          <div className="header-status">Online</div>
          <button className="dark-toggle" onClick={toggleDarkMode} aria-label="Toggle dark mode">
            {isDarkMode ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      <main className="main">
        <div className="chat-messages">
          {messages.length === 0 && welcomeTyping !== '' && (
            <div className="message-wrapper assistant">
              <div className="message assistant">
                <div className="avatar assistant">{currentAvatar}</div>
                <div className="message-content">
                  <p className="welcome-text" dangerouslySetInnerHTML={{ __html: welcomeTyping.replace(/\n/g, '<br/>') }} />
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message-wrapper ${msg.role}`}>
              <div className={`message ${msg.role}`}>
                <div className={`avatar ${msg.role}`}>{msg.role === 'user' ? 'U' : currentAvatar}</div>
                <div className="message-content">
                  {msg.image && <img src={msg.image} alt="Uploaded" className="uploaded-image" />}
                  <p dangerouslySetInnerHTML={{ __html: msg.isTyping ? (msg.content || '') : (msg.content || '').replace(/\n/g, '<br/>') }} />
                  {msg.hasMemory && !msg.isTyping && <span className="memory-icon">🧠</span>}
                  {!msg.isTyping && <div className="message-time">{msg.timestamp}</div>}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-wrapper assistant">
              <div className="message assistant">
                <div className="avatar assistant">{currentAvatar}</div>
                <div className="message-content typing-indicator">
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
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
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            disabled={loading}
            className="input-field"
            rows={1}
          />
          <button onClick={sendMessage} disabled={loading || (!input.trim() && !image)} className="send-btn">
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
