(function() {
    'use strict';
    const CONFIG = {
        API_BASE_URL: 'http://localhost:8000',
        THEME: 'engineering',
        POSITION: 'bottom-right',
        AUTO_SHOW_NOTIFICATION: true,
        NOTIFICATION_DELAY: 5000,
        CONVERSATION_MEMORY: true,
        TYPING_DELAY: 1200,
        MAX_MESSAGE_LENGTH: 500,
        WIDGET_TITLE: "LIBRO",
        WELCOME_MESSAGE_DURATION: 3000
    };
    let isOpen = false,
        isTyping = false,
        sessionId = 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
        conversationHistory = [],
        messageCount = 0,
        hasShownNotification = false;

    // ===== Theme / Utility =====
    function getThemeColors(theme) {
        return {
            engineering: {
                primary: '#2c3e50',
                secondary: '#3498db',
                light: '#ecf0f1',
                accent: '#e74c3c',
                text: '#2c3e50',
                gradient: 'linear-gradient(135deg, #3498db 0%, #2c3e50 100%)',
                shadow: 'rgba(44, 62, 80, 0.3)'
            }
        }[theme] || {};
    }
    function getPositionStyles(position) {
        return {
            'bottom-right': { bottom: '20px', right: '20px' },
            'bottom-left': { bottom: '20px', left: '20px' },
            'top-right': { top: '20px', right: '20px' },
            'top-left': { top: '20px', left: '20px' }
        }[position];
    }
    function isMobile() { return window.innerWidth <= 768; }

    // ============= Styles ============
    function createStyles() {
        const c = getThemeColors(CONFIG.THEME),
              p = getPositionStyles(CONFIG.POSITION);
        const css = `
#msrit-chatbot *,
#msrit-chatbot *:focus { box-sizing:border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; outline:none; }
#msrit-chatbot { position:fixed; ${Object.entries(p).map(([k, v]) => `${k}:${v}`).join(';')}; z-index:2147483647; font-size:14px; }
#msrit-chatbot .chat-toggle { width:64px; height:64px; border-radius:50%; border:none; cursor:pointer; background:${c.gradient}; color:#fff; font-size:26px; box-shadow:0 8px 20px ${c.shadow}; transition:.3s; display:flex; align-items: center; justify-content: center; }
#msrit-chatbot .chat-toggle.open { background: ${c.accent}; }
#msrit-chatbot .chat-notification { position:absolute; top:6px; right:6px; width:20px;height:20px;background:${c.accent};border-radius:50%;display:none;align-items:center;justify-content:center;font-size:13px; color:#fff;border:2px solid #fff;}
#msrit-chatbot .chat-notification.show { display:flex; }
.welcome-popup { position:fixed; z-index:2147483646; bottom:90px; right:20px; background:${c.primary}; color:#fff; padding:15px; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.15); max-width:280px; opacity:0; transform:translateY(20px); transition:all .4s; font-size:14px; }
.welcome-popup.show { opacity:1; transform:translateY(0);}
@media (max-width:768px) { .welcome-popup{ max-width:85vw; bottom:80px; right:20px; font-size:13px; } }
.chat-window { position:absolute; bottom:70px; right:0; width:360px; height:540px; background:#fff; border-radius:8px; box-shadow:0 20px 50px rgba(0,0,0,.15); display:none; flex-direction:column; transform:scale(.97); opacity:0; transition:.2s; overflow:hidden; border:1px solid rgba(0,0,0,.08);}
.chat-window.open { display:flex; transform:scale(1); opacity:1; }
@media (max-width:768px) { .chat-window{ width:100vw; height:99vh; border-radius:0; left:0; right:0; bottom:0; top:0;} }
.chat-header { background:${c.gradient}; color:#fff; padding:12px 18px; border-radius:8px 8px 0 0; display:flex; align-items:center; justify-content:space-between; min-height:48px;}
.chat-header h3 { flex:1; text-align:center; font-size:16px; font-weight:600; letter-spacing:.02em;}
.chat-header-buttons { display: flex; gap: 10px; }
.chat-header button { background:none; border:none; color:#fff; font-size:20px; cursor:pointer; width:28px; height:28px; border-radius:50%; transition:background 0.2s; }
.chat-header button:focus, .chat-header button:hover { background:rgba(255,255,255,.18); }
.chat-messages { flex:1; padding:16px; overflow-y:auto; background:${c.light}; }
.message { display:flex; margin-bottom:16px; align-items:flex-end;}
.message.user { justify-content:flex-end; }
.message.bot .message-avatar { background:${c.gradient}; color:#fff; }
.message.user .message-avatar { background:#ddd; color:${c.text}; }
.message-avatar { width:36px; height:36px; border-radius:50%; margin:0 10px; font-size:16px; display:flex; align-items:center; justify-content:center; }
.message-bubble { max-width:84%; padding:12px 16px; border-radius:18px; font-size:14px; box-shadow:0 2.5px 14px rgba(0,0,0,.07); outline: none;}
.message.user .message-bubble { background:${c.gradient}; color:#fff; border-bottom-right-radius:4px; border:0;}
.message.bot .message-bubble { background:#fff; color:${c.text}; border:1px solid rgba(0,0,0,.08); border-bottom-left-radius:4px;}
.message-bubble:focus { box-shadow:0 0 0 2px ${c.accent}; }
.typing-indicator { display:none; padding:10px 18px; background:#fff; border-radius:20px 20px 10px 20px; border:1px solid #ddd; box-shadow:0 1px 2px rgba(0,0,0,.07); margin-bottom:10px; margin-left:8px; font-size:14px; }
.typing-indicator.active { display:block;}
.typing-indicator[aria-live] { position:absolute; left:-9999px; height:1px; width:1px; overflow:hidden; }
.typing-dots { display:inline-flex; gap:5px; align-items:center; }
.typing-dots span { width:8px; height:8px; background:${c.accent}; border-radius:50%; opacity:.6; animation:typingDot 1150ms infinite alternate;}
.typing-dots span:nth-child(2) { animation-delay:.2s; }
.typing-dots span:nth-child(3) { animation-delay:.4s; }
@keyframes typingDot { 0%,100%{opacity:.4; transform:translateY(0);} 40%{opacity:1; transform:translateY(-3px);} }
.chat-input { padding:12px; background:#fff; border-top:1px solid #ddd; display:flex; gap:8px; }
.chat-input input { flex:1; padding:12px 16px; border:1.5px solid #ddd; border-radius: 24px; font-size:14px; background:${c.light}; outline:none;}
.chat-input input:focus { border-color:${c.accent}; box-shadow:0 0 0 2px rgba(231, 76, 60, 0.12);}
.send-btn { width:40px; height:40px; border-radius:50%; background:${c.primary}; color:#fff; border:none; font-size:18px; cursor:pointer; transition:.2s;}
.send-btn:disabled { background:#ccc; cursor:not-allowed;}
.send-btn:focus, .send-btn:hover { background:${c.secondary}; }
@media (max-width:768px) { .chat-input {padding:10px 7px;} }
.error-msg { background:#ffebee; color:#c62828; padding:10px 15px; border-radius:4px; margin:10px 0; font-size:13px; border:1px solid #ef9a9a; }
        `;
        const style = document.createElement('style');
        style.id = 'msrit-chatbot-styles';
        style.textContent = css;
        const existingStyle = document.querySelector('#msrit-chatbot-styles');
        if (existingStyle) existingStyle.remove();
        document.head.appendChild(style);
    }

    // ============ Markup/Template =============
    function createWidget() {
        const widget = document.createElement('div');
        widget.id = 'msrit-chatbot';
        widget.setAttribute('aria-hidden', 'false');
        widget.innerHTML = `
            <div class="welcome-popup" id="welcome-popup" role="status" aria-live="polite" aria-atomic="true">
                Hello! I'm here to assist you.
            </div>
            <button class="chat-toggle" id="chat-toggle" title="Open chat window" aria-haspopup="dialog" aria-label="Open chat window" tabindex="0">
                <span id="toggle-icon">💬</span>
                <div class="chat-notification" id="notification" aria-label="You have a message">!</div>
            </button>
            <div class="chat-window" id="chat-window" role="dialog" aria-modal="true" aria-labelledby="ada-title" tabindex="-1">
                <div class="chat-header">
                    <h3 id="ada-title">${CONFIG.WIDGET_TITLE}</h3>
                    <div class="chat-header-buttons">
                        <button class="new-chat-btn" id="new-chat-btn" aria-label="Start new chat" tabindex="0">+</button>
                        <button class="minimize-btn" id="minimize-btn" aria-label="Minimize chat" tabindex="0">−</button>
                    </div>
                </div>
                <div class="chat-messages" id="chat-messages" aria-live="polite" aria-atomic="false" tabindex="0"></div>
                <div class="typing-indicator" id="typing-indicator" aria-live="polite" aria-hidden="true">
                    <span class="sr-only" id="typing-announce">Engineering Assistant is typing…</span>
                    <div class="typing-dots"><span></span><span></span><span></span></div>
                </div>
                <form class="chat-input" id="chat-input-form" autocomplete="off" aria-label="Send message to Engineering Assistant">
                    <input type="text" id="chat-input"
                        placeholder="Type your query..."
                        maxlength="${CONFIG.MAX_MESSAGE_LENGTH}"
                        aria-label="Type your engineering question"
                        autocomplete="off"
                        required
                    >
                    <button class="send-btn" id="send-btn" title="Send message" aria-label="Send message" tabindex="0" type="submit">➤</button>
                </form>
            </div>
        `;
        document.body.appendChild(widget);
        return {
            chatToggle:   widget.querySelector('#chat-toggle'),
            chatWindow:   widget.querySelector('#chat-window'),
            chatInput:    widget.querySelector('#chat-input'),
            sendBtn:      widget.querySelector('#send-btn'),
            newChatBtn:   widget.querySelector('#new-chat-btn'),
            minimizeBtn:  widget.querySelector('#minimize-btn'),
            chatMessages: widget.querySelector('#chat-messages'),
            typingInd:    widget.querySelector('#typing-indicator'),
            welcomePopup: widget.querySelector('#welcome-popup'),
            notification: widget.querySelector('#notification'),
            inputForm:    widget.querySelector('#chat-input-form')
        };
    }

    // ============ Message / Fallbacks =============
    function formatMessage(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>')
            .replace(/([\w\.-]+@[\w\.-]+\.\w+)/g, '<a href="mailto:$1">$1</a>')
            .replace(/(\d{3}-\d{4}-\d{4})/g, '<a href="tel:$1">$1</a>');
    }

    // ============ Logic ============
    let els = {};
    function addMessage(content, sender, focus) {
        const div = document.createElement('div');
        div.className = `message ${sender}`;
        let innerHTML;
        if (sender === "bot") {
            innerHTML = `
                <div class="message-avatar" aria-hidden="true">🤖</div>
                <div class="message-bubble" tabindex="0" role="group" aria-label="Message from Engineering Assistant">${formatMessage(content)}</div>
            `;
        } else {
            innerHTML = `
                <div class="message-bubble" tabindex="0" role="group" aria-label="Message from user">${formatMessage(content)}</div>
                <div class="message-avatar" aria-hidden="true">👤</div>
            `;
        }
        div.innerHTML = innerHTML;
        els.chatMessages.appendChild(div);
        els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
        if (CONFIG.CONVERSATION_MEMORY) {
            conversationHistory.push({ content, sender, timestamp: new Date().toISOString() });
            if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);
        }
        if (focus) {
            const bubbles = div.querySelectorAll('.message-bubble');
            if (bubbles && bubbles.length) bubbles[0].focus();
        }
    }
    function showTyping(show) {
        if (show) {
            els.typingInd.setAttribute('aria-hidden', "false");
            els.typingInd.classList.add('active');
            if (!els.typingInd.parentNode || els.typingInd.parentNode !== els.chatMessages)
                els.chatMessages.appendChild(els.typingInd);
            els.typingInd.querySelector('.sr-only').textContent = "Engineering Assistant is typing…";
            els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
        } else {
            els.typingInd.classList.remove('active');
            els.typingInd.setAttribute('aria-hidden', "true");
            els.typingInd.querySelector('.sr-only').textContent = "";
        }
    }
    function showWelcomeMessage() {
        if (els.welcomePopup) {
            setTimeout(() => { els.welcomePopup.classList.add("show"); }, 500);
            setTimeout(() => {
                els.welcomePopup.classList.remove("show");
                setTimeout(() => els.welcomePopup.remove(), 600);
            }, CONFIG.WELCOME_MESSAGE_DURATION);
        }
    }
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = "error-msg";
        errorDiv.setAttribute("role", "alert");
        errorDiv.innerHTML = "⚠️ " + message;
        els.chatMessages.appendChild(errorDiv);
        els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
        setTimeout(() => { if (errorDiv.parentNode) errorDiv.remove(); }, 4500);
    }
    async function sendToAPI(message) {
        showTyping(true);
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/ask`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data.answer;
        } catch (error) {
            console.error("Error calling API:", error);
            throw error;
        }
    }
    async function sendMessage(message = null) {
        const text = message || (els.chatInput?.value || '').trim();
        if (!text || isTyping) return;
        addMessage(text, 'user', false);
        if (!message && els.chatInput) els.chatInput.value = '';
        isTyping = true;
        updateInputState();
        showTyping(true);
        try {
            const response = await sendToAPI(text);
            showTyping(false);
            isTyping = false;
            addMessage(response, 'bot', true);
        } catch (error) {
            showTyping(false);
            isTyping = false;
            showError("Sorry, I couldn't process your request. Please try again.");
        }
        updateInputState();
    }
    function updateInputState() {
        if (els.chatInput && els.sendBtn) {
            const hasText = els.chatInput.value.trim().length > 0;
            els.sendBtn.disabled = !hasText || isTyping;
            els.chatInput.disabled = isTyping;
        }
    }
    function toggleChat(open) {
        if (!els.chatWindow || !els.chatToggle) return;
        if (typeof open !== "boolean") open = !isOpen;
        els.chatWindow.classList.toggle('open', open);
        els.chatToggle.classList.toggle('open', open);
        document.getElementById('toggle-icon').textContent = open ? '✕' : '💬';
        isOpen = open;
        if (open) {
            if (els.notification) els.notification.classList.remove('show');
            setTimeout(() => { if (els.chatInput) els.chatInput.focus(); }, 300);
            trapFocus();
        } else {
            els.chatToggle.focus();
            untrapFocus();
        }
    }
    function newChat() {
        els.chatMessages.innerHTML = '';
        conversationHistory = [];
        addMessage(
            `<strong>Hello! I'm your Virtual Library Assistant.</strong><br><br>I can help you with:<br>
            • Engineering resources and technical papers<br>• Project and research support<br>• Career and placement advice<br><br>How can I assist you today?`,
            'bot',
            true
        );
    }
    function showNotification() {
        if (!isOpen && CONFIG.AUTO_SHOW_NOTIFICATION && !hasShownNotification && els.notification) {
            els.notification.classList.add('show');
            hasShownNotification = true;
            setTimeout(() => { els.notification.classList.remove('show'); }, 9000);
        }
    }
    let lastActiveBeforeOpen = null;
    function trapFocus() {
        const focusableSelectors = 'button, [href], input:not([disabled])';
        const keyboardEls = Array.from(els.chatWindow.querySelectorAll(focusableSelectors)).filter(el => el.offsetParent != null);
        let first = keyboardEls[0], last = keyboardEls[keyboardEls.length - 1];
        lastActiveBeforeOpen = document.activeElement;
        els.chatWindow.addEventListener('keydown', focusTrapHandler);
        function focusTrapHandler(e) {
            if (e.key === "Tab") {
                if (e.shiftKey) {
                    if (document.activeElement === first) { last.focus(); e.preventDefault(); }
                } else if (document.activeElement === last) { first.focus(); e.preventDefault(); }
            }
            if (e.key === "Escape") {
                e.preventDefault(); toggleChat(false);
            }
        }
    }
    function untrapFocus() {
        els.chatWindow.removeEventListener('keydown', focusTrapHandler);
        if (lastActiveBeforeOpen) lastActiveBeforeOpen.focus();
    }
    function setupEvents() {
        els.chatToggle.addEventListener('click', () => toggleChat());
        els.minimizeBtn.addEventListener('click', () => toggleChat(false));
        els.newChatBtn.addEventListener('click', newChat);
        els.inputForm.addEventListener('submit', e => {
            e.preventDefault();
            sendMessage();
        });
        els.chatInput.addEventListener('input', updateInputState);
        els.chatInput.addEventListener('keydown', e => {
            if (e.key == "Escape") { toggleChat(false); }
        });
        window.addEventListener('keydown', e => {
            if (!isOpen && (e.key === " " || e.key === "Enter") && document.activeElement === els.chatToggle)
                toggleChat(true);
        });
    }
    function init() {
        if (document.getElementById('msrit-chatbot')) return;
        createStyles();
        els = createWidget();
        setupEvents();
        updateInputState();
        showWelcomeMessage();
        newChat();
        if (CONFIG.AUTO_SHOW_NOTIFICATION && CONFIG.NOTIFICATION_DELAY > 0)
            setTimeout(showNotification, CONFIG.NOTIFICATION_DELAY);
    }
    window.EngineeringAssistant = {
        open: () => !isOpen && toggleChat(true),
        close: () => isOpen && toggleChat(false),
        toggle: () => toggleChat(),
        send: msg => sendMessage(msg),
        newChat: () => newChat(),
        isOpen: () => isOpen
    };
    (document.readyState === "loading") ? document.addEventListener("DOMContentLoaded", init) : setTimeout(init, 75);
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === "visible" && isOpen)
            setTimeout(() => { if (els.chatInput) els.chatInput.focus(); }, 120);
    });
})();
