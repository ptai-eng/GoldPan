"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  UploadCloud, Link as LinkIcon, Loader2, FileText, 
  X, Sparkles, LayoutTemplate, Database, 
  Settings, ChevronRight, CheckCircle2, Download, Clock, Menu
} from "lucide-react";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"file" | "url">("file");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [result, setResult] = useState<{ markdown: string; metadata: any } | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [apiKey, setApiKey] = useState("");
  
  // Settings State
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const [tempApiKey, setTempApiKey] = useState("");
  const [isKeyValid, setIsKeyValid] = useState(false);

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light") {
      setTheme("light");
      document.documentElement.classList.add("light");
    }
    const savedZoom = localStorage.getItem("zoomLevel");
    if (savedZoom) {
      const parsedZoom = parseFloat(savedZoom);
      setZoomLevel(parsedZoom);
      document.documentElement.style.fontSize = `${parsedZoom * 16}px`;
    }
  }, []);

  const changeTheme = (newTheme: "dark" | "light") => {
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
    if (newTheme === "light") {
      document.documentElement.classList.add("light");
    } else {
      document.documentElement.classList.remove("light");
    }
  };

  const changeZoom = (newZoom: number) => {
    setZoomLevel(newZoom);
    localStorage.setItem("zoomLevel", newZoom.toString());
    document.documentElement.style.fontSize = `${newZoom * 16}px`;
  };
  // Persist API Key
  useEffect(() => {
    const savedKey = localStorage.getItem("goldpan_api_key");
    if (savedKey) {
      setApiKey(savedKey);
      setTempApiKey(savedKey);
      if (savedKey.startsWith("AIza") && savedKey.length > 30) {
        setIsKeyValid(true);
      }
    }
  }, []);

  useEffect(() => {
    if (apiKey) {
      localStorage.setItem("goldpan_api_key", apiKey);
    } else {
      localStorage.removeItem("goldpan_api_key");
    }
  }, [apiKey]);

  const handleConfirmApiKey = () => {
    if (tempApiKey.trim() === "") {
      setApiKey("");
      setIsKeyValid(false);
      return;
    }
    if (tempApiKey.startsWith("AIza") && tempApiKey.length > 30) {
      setApiKey(tempApiKey);
      setIsKeyValid(true);
    } else {
      alert("Invalid Gemini API Key format. It should start with 'AIza'.");
      setIsKeyValid(false);
    }
  };
  
  // App views
  const [activeView, setActiveView] = useState<"extractor" | "kb">("extractor");
  const [kbDocuments, setKbDocuments] = useState<any[]>([]);
  const [chatHistory, setChatHistory] = useState<{role: "user" | "bot", content: string}[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isSavingKb, setIsSavingKb] = useState(false);

  // Timer states
  const [estimatedSeconds, setEstimatedSeconds] = useState(0);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Timer countdown logic
  useEffect(() => {
    if (status === 'loading' && remainingSeconds > 0) {
      const timer = setTimeout(() => {
        // Dừng ở 1 giây nếu backend chưa xử lý xong để tránh bị nhảy qua số âm
        if (remainingSeconds > 1) {
          setRemainingSeconds(r => r - 1);
        }
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [status, remainingSeconds]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const calculateEstimate = (fileSize: number) => {
    // Ước tính: ~10 giây cho mỗi MB dữ liệu
    const sizeMB = fileSize / (1024 * 1024);
    // Tối thiểu 5 giây, các file siêu nặng có thể lên tới vài phút
    return Math.max(5, Math.ceil(sizeMB * 10));
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const processUrl = async () => {
    if (!url) return;
    
    // URL thường nhanh hơn, ước tính 8 giây
    setEstimatedSeconds(8);
    setRemainingSeconds(8);
    setStatus("loading");
    
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (apiKey) headers["X-Gemini-Api-Key"] = apiKey;

      const res = await fetch("http://localhost:8000/api/extract/url", {
        method: "POST",
        headers,
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to process URL");
      
      setRemainingSeconds(0); // Force về 0 khi xong
      setResult(data);
      setStatus("success");
    } catch (err: any) {
      setErrorMessage(err.message);
      setStatus("error");
    }
  };

  const processFile = async () => {
    if (!file) return;
    
    if (file.size > 1024 * 1024 && !apiKey) {
      const proceed = window.confirm("⚠️ Large Document Detected (>1MB)\n\nSince no Gemini API Key is provided, the system will fallback to basic Local Regex filtering, which may result in lower quality cleaning.\n\nDo you want to proceed with Local processing?");
      if (!proceed) return;
    }
    
    // Tính toán thời gian dựa trên dung lượng file
    const est = calculateEstimate(file.size);
    setEstimatedSeconds(est);
    setRemainingSeconds(est);
    setStatus("loading");
    
    const formData = new FormData();
    formData.append("file", file);
    try {
      const headers: Record<string, string> = {};
      if (apiKey) headers["X-Gemini-Api-Key"] = apiKey;

      const res = await fetch("http://localhost:8000/api/extract/file", {
        method: "POST",
        headers,
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to process file");
      
      setRemainingSeconds(0); // Force về 0 khi xong
      setResult(data);
      setStatus("success");
    } catch (err: any) {
      setErrorMessage(err.message);
      setStatus("error");
    }
  };

  const reset = () => {
    setStatus("idle");
    setResult(null);
    setFile(null);
    setUrl("");
    setRemainingSeconds(0);
    setEstimatedSeconds(0);
  };

  const saveToKb = async () => {
    if (!result) return;
    setIsSavingKb(true);
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (apiKey) headers["X-Gemini-Api-Key"] = apiKey;
      
      const res = await fetch("http://localhost:8000/api/kb/ingest", {
        method: "POST",
        headers,
        body: JSON.stringify({ markdown: result.markdown, metadata: result.metadata }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to save to KB");
      }
      alert("Successfully saved to Knowledge Base!");
    } catch (err: any) {
      alert("Error saving to KB: " + err.message);
    } finally {
      setIsSavingKb(false);
    }
  };

  const fetchKbDocuments = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/kb/documents");
      const data = await res.json();
      if (res.ok) {
        setKbDocuments(data.documents || []);
      }
    } catch (err) {
      console.error("Failed to fetch docs", err);
    }
  };

  const handleSendChat = async () => {
    if (!chatInput.trim() || isChatLoading) return;
    
    const userMsg = chatInput.trim();
    setChatInput("");
    setChatHistory(prev => [...prev, { role: "user", content: userMsg }]);
    setIsChatLoading(true);

    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (apiKey) headers["X-Gemini-Api-Key"] = apiKey;

      const res = await fetch("http://localhost:8000/api/kb/chat", {
        method: "POST",
        headers,
        body: JSON.stringify({ message: userMsg, history: chatHistory }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Chat failed");
      
      setChatHistory(prev => [...prev, { role: "bot", content: data.response }]);
    } catch (err: any) {
      setChatHistory(prev => [...prev, { role: "bot", content: `**Error:** ${err.message}` }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const downloadMarkdown = () => {
    if (!result) return;
    const blob = new Blob([result.markdown], { type: 'text/markdown' });
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = result.metadata.title ? `${result.metadata.title.replace(/\.[^/.]+$/, "")}.md` : 'extracted-goldpan.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(blobUrl);
  };

  // Tính phần trăm chạy của progress bar
  const progressPercent = estimatedSeconds > 0 
    ? Math.min(100, Math.max(0, ((estimatedSeconds - remainingSeconds) / estimatedSeconds) * 100)) 
    : 0;

  return (
    <div className="min-h-screen bg-background text-foreground flex overflow-hidden selection:bg-accent selection:text-primary">
      
      {/* Sidebar */}
      <aside className={`border-r border-border bg-panel hidden md:flex flex-col transition-[width,opacity] duration-300 ease-in-out overflow-hidden ${isSidebarOpen ? 'w-64 opacity-100' : 'w-0 opacity-0 border-r-0'}`}>
        <div className="w-64 flex flex-col h-full">
          <div className="p-6 flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-gradient-to-tr from-muted to-accent flex items-center justify-center border border-panel-border shadow-lg">
            <Sparkles className="w-4 h-4 text-primary" />
          </div>
          <span className="font-semibold text-primary tracking-wide text-sm">GoldPan AI</span>
        </div>

        <nav className="flex-1 px-4 py-4 flex flex-col gap-1">
          <div className="px-3 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Workspace</div>
          <button 
            onClick={() => setActiveView("extractor")}
            className={`flex items-center gap-3 px-3 py-2 text-sm w-full text-left rounded-md transition-colors ${activeView === 'extractor' ? 'bg-muted/50 text-primary border border-panel-border shadow-inner' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'}`}
          >
            <LayoutTemplate className="w-4 h-4" />
            Extractor Engine
          </button>
          <button 
            onClick={() => { setActiveView("kb"); fetchKbDocuments(); }}
            className={`flex items-center gap-3 px-3 py-2 text-sm w-full text-left rounded-md transition-colors ${activeView === 'kb' ? 'bg-muted/50 text-primary border border-panel-border shadow-inner' : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'}`}
          >
            <Database className="w-4 h-4" />
            Knowledge Base
          </button>
        </nav>

        <div className="p-4 border-t border-border">
          <button onClick={() => setIsSettingsOpen(true)} className="w-full flex items-center gap-3 px-3 py-2 text-sm text-muted-foreground hover:text-foreground rounded-md transition-colors">
            <Settings className="w-4 h-4" />
            Settings
          </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative h-screen overflow-hidden">
        {/* Glow Effect */}
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-accent/20 blur-[120px] pointer-events-none" />

        <header className="h-16 border-b border-border/50 flex items-center px-4 bg-background/80 backdrop-blur-md z-10 gap-4">
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-md transition-colors"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Workspace</span>
            <ChevronRight className="w-4 h-4" />
            <span className="text-foreground">{activeView === 'extractor' ? 'Data Cleansing Pipeline' : 'Knowledge Base Chat'}</span>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          {activeView === "extractor" ? (
            <>
              {/* Left Panel: Controls */}
              <div className="w-full lg:w-1/3 border-r border-border/50 bg-panel/30 p-8 overflow-y-auto flex flex-col gap-8 z-10">
                
                <div>
                  <h1 className="text-2xl font-semibold text-primary mb-2">Extract Content</h1>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Transform noisy PDFs and raw HTML into clean, AI-ready Markdown.
                  </p>
                </div>

                <div className="bg-panel border border-panel-border/80 rounded-xl p-1 flex shadow-sm">
                  <button 
                    onClick={() => setActiveTab("file")}
                    className={`flex-1 py-2.5 text-sm font-medium rounded-lg transition duration-200 ${activeTab === 'file' ? 'bg-accent text-primary shadow-md border border-panel-border/50' : 'text-muted-foreground hover:text-foreground'}`}
                  >
                    Document
                  </button>
                  <button 
                    onClick={() => setActiveTab("url")}
                    className={`flex-1 py-2.5 text-sm font-medium rounded-lg transition duration-200 ${activeTab === 'url' ? 'bg-accent text-primary shadow-md border border-panel-border/50' : 'text-muted-foreground hover:text-foreground'}`}
                  >
                    Web Link
                  </button>
                </div>

                <AnimatePresence mode="wait">
                  {activeTab === "file" ? (
                    <motion.div
                      key="file-tab"
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -5 }}
                      className="flex flex-col gap-6"
                    >
                      <div 
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                        className={`relative overflow-hidden border border-dashed rounded-2xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition duration-300 ${
                          isDragging 
                            ? "border-zinc-500 bg-muted/80 shadow-[0_0_30px_rgba(255,255,255,0.05)]" 
                            : "border-panel-border bg-panel hover:bg-[#0f0f11] hover:border-panel-border"
                        } ${file ? "border-green-500/30 bg-green-950/20" : ""}`}
                      >
                        <input 
                          type="file" 
                          className="hidden" 
                          ref={fileInputRef} 
                          onChange={handleFileSelect} 
                        />
                        
                        {file ? (
                          <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} className="flex flex-col items-center">
                            <div className="w-12 h-12 bg-green-500/10 rounded-full flex items-center justify-center mb-4 border border-green-500/20">
                              <CheckCircle2 className="w-6 h-6 text-green-500" />
                            </div>
                            <p className="text-primary font-medium text-sm">{file.name}</p>
                            <p className="text-muted-foreground text-xs mt-1.5 font-mono">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                          </motion.div>
                        ) : (
                          <>
                            <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center mb-4 border border-panel-border">
                              <UploadCloud className="w-5 h-5 text-muted-foreground" />
                            </div>
                            <p className="text-foreground font-medium text-sm">Click or drag document here</p>
                            <p className="text-muted-foreground text-xs mt-2 font-medium tracking-wide">PDF, DOCX, TXT, JPG, MP3</p>
                          </>
                        )}
                      </div>

                      <button 
                        onClick={processFile}
                        disabled={!file || status === "loading"}
                        className="group relative w-full bg-white text-black font-semibold py-3.5 rounded-xl transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center overflow-hidden hover:bg-zinc-200 shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)]"
                      >
                        {status === "loading" ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <span className="flex items-center gap-2">
                            Start Extraction <Sparkles className="w-4 h-4 opacity-70 group-hover:opacity-100 transition-opacity" />
                          </span>
                        )}
                      </button>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="url-tab"
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -5 }}
                      className="flex flex-col gap-6"
                    >
                      <div className="bg-panel border border-panel-border rounded-xl overflow-hidden focus-within:border-zinc-600 focus-within:ring-1 focus-within:ring-zinc-600 transition">
                        <div className="flex items-center gap-3 p-4">
                          <LinkIcon className="w-4 h-4 text-muted-foreground" />
                          <input 
                            type="url" 
                            placeholder="https://example.com/article" 
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            className="bg-transparent border-none outline-none text-sm text-primary w-full placeholder:text-muted-foreground font-mono"
                          />
                        </div>
                      </div>
                      
                      <button 
                        onClick={processUrl}
                        disabled={!url || status === "loading"}
                        className="group relative w-full bg-white text-black font-semibold py-3.5 rounded-xl transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center overflow-hidden hover:bg-zinc-200"
                      >
                        {status === "loading" ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <span className="flex items-center gap-2">
                            Start Extraction <Sparkles className="w-4 h-4 opacity-70 group-hover:opacity-100 transition-opacity" />
                          </span>
                        )}
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>

                {status === "error" && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }} 
                    animate={{ opacity: 1, scale: 1 }} 
                    className="bg-red-950/50 border border-red-900/50 text-red-400 p-4 rounded-xl text-sm font-medium flex items-start gap-3"
                  >
                    <X className="w-5 h-5 shrink-0 mt-0.5 opacity-70" />
                    <p className="leading-relaxed">{errorMessage}</p>
                  </motion.div>
                )}
              </div>

          {/* Right Panel: Output */}
          <div className="flex-1 bg-panel p-8 overflow-y-auto relative z-10 flex flex-col">
            <div className="flex-1 border border-border bg-panel/50 rounded-2xl shadow-2xl flex flex-col overflow-hidden backdrop-blur-sm">
              <div className="h-14 border-b border-border flex items-center justify-between px-4 bg-panel">
                <div className="flex items-center gap-4">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-accent"></div>
                    <div className="w-3 h-3 rounded-full bg-accent"></div>
                    <div className="w-3 h-3 rounded-full bg-accent"></div>
                  </div>
                  <div className="w-px h-5 bg-accent mx-1"></div>
                  <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider font-semibold">Markdown Output</span>
                </div>
                
                {/* Actions: Download & Clear */}
                {result && (
                  <div className="flex items-center gap-3">
                    <button 
                      onClick={saveToKb} 
                      disabled={isSavingKb}
                      className="flex items-center gap-2 text-primary hover:text-black hover:bg-white transition-colors text-xs font-semibold bg-blue-900/50 px-3 py-1.5 rounded-md border border-blue-700 disabled:opacity-50"
                    >
                      {isSavingKb ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Database className="w-3.5 h-3.5" />}
                      Save to KB
                    </button>
                    <button 
                      onClick={downloadMarkdown} 
                      className="flex items-center gap-2 text-primary hover:text-black hover:bg-white transition-colors text-xs font-semibold bg-accent px-3 py-1.5 rounded-md border border-panel-border"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download MD
                    </button>
                    <button 
                      onClick={reset} 
                      className="text-muted-foreground hover:text-red-400 transition-colors"
                      title="Clear Workspace"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                )}
              </div>

              <div className="p-8 flex-1 overflow-y-auto font-sans">
                {status === "idle" && (
                  <div className="h-full flex flex-col items-center justify-center text-muted-foreground gap-4">
                    <FileText className="w-12 h-12 opacity-20" />
                    <p className="text-sm font-medium tracking-wide">Ready to extract some gold.</p>
                  </div>
                )}
                
                {status === "loading" && (
                  <div className="h-full flex flex-col items-center justify-center gap-8 text-muted-foreground w-full max-w-md mx-auto">
                    
                    <div className="flex flex-col items-center gap-2 relative">
                      <Clock className="w-8 h-8 text-muted-foreground mb-2 animate-pulse" />
                      <div className="text-3xl font-mono text-primary font-light tracking-widest">
                        {formatTime(remainingSeconds)}
                      </div>
                      <p className="text-xs text-muted-foreground uppercase tracking-widest">Estimated Time Remaining</p>
                      
                      <AnimatePresence>
                        {remainingSeconds === 1 && (
                          <motion.div 
                            initial={{ opacity: 0, y: 5 }} 
                            animate={{ opacity: 1, y: 0 }} 
                            className="absolute -bottom-10 whitespace-nowrap text-xs font-medium text-red-400/80 bg-red-950/40 px-3 py-1.5 rounded-full border border-red-900/50 flex items-center gap-2"
                          >
                            <Loader2 className="w-3 h-3 animate-spin" />
                            <span>Processing complex data... Please wait a moment.</span>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>

                    {/* Progress Bar UI */}
                    <div className="w-full flex flex-col gap-2 mt-4">
                      <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-white rounded-full"
                          initial={{ width: "0%" }}
                          animate={{ width: `${progressPercent}%` }}
                          transition={{ ease: "linear", duration: 1 }}
                        />
                      </div>
                      <div className="flex justify-between text-xs font-mono text-muted-foreground">
                        <span>Extracting text...</span>
                        <span>{Math.round(progressPercent)}%</span>
                      </div>
                    </div>
                  </div>
                )}
                
                {status === "success" && result && (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                    className="markdown-body max-w-3xl mx-auto"
                  >
                    <ReactMarkdown>{result.markdown}</ReactMarkdown>
                  </motion.div>
                )}
              </div>
            </div>
          </div>
            </>
          ) : (
            <>
              {/* Left Panel: List of Docs in KB */}
              <div className="w-full lg:w-1/3 border-r border-border/50 bg-panel/30 p-8 overflow-y-auto flex flex-col gap-8 z-10">
                <div>
                  <h1 className="text-2xl font-semibold text-primary mb-2">Knowledge Base</h1>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Documents available for AI querying.
                  </p>
                </div>

                <div className="flex flex-col gap-3">
                  {kbDocuments.length === 0 ? (
                    <div className="text-muted-foreground text-sm italic">No documents found. Extract something and click 'Save to KB'.</div>
                  ) : (
                    kbDocuments.map((doc, idx) => (
                      <div key={idx} className="bg-panel border border-panel-border p-4 rounded-xl flex items-start gap-3 hover:border-panel-border transition-colors">
                        <FileText className="w-5 h-5 text-muted-foreground mt-0.5 shrink-0" />
                        <div>
                          <p className="text-sm font-medium text-foreground line-clamp-1">{doc.title}</p>
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-1 font-mono">{doc.source}</p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                
                <button onClick={fetchKbDocuments} className="text-xs text-muted-foreground hover:text-foreground text-left underline underline-offset-2">Refresh List</button>
              </div>

              {/* Right Panel: Chat UI */}
              <div className="flex-1 bg-panel p-8 relative z-10 flex flex-col">
                <div className="flex-1 border border-border bg-panel/50 rounded-2xl shadow-2xl flex flex-col overflow-hidden backdrop-blur-sm">
                  <div className="h-14 border-b border-border flex items-center justify-between px-4 bg-panel">
                    <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider font-semibold">GoldPan AI Assistant</span>
                  </div>

                  <div className="flex-1 p-6 overflow-y-auto flex flex-col gap-6 font-sans">
                    {chatHistory.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-muted-foreground gap-4">
                        <Sparkles className="w-12 h-12 opacity-20" />
                        <p className="text-sm font-medium tracking-wide">Ask anything about your Knowledge Base.</p>
                      </div>
                    ) : (
                      chatHistory.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[80%] rounded-2xl px-5 py-3.5 ${msg.role === 'user' ? 'bg-accent text-foreground' : 'bg-transparent border border-panel-border text-foreground'}`}>
                            {msg.role === 'user' ? (
                              msg.content
                            ) : (
                              <div className="markdown-body text-sm">
                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                              </div>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                    {isChatLoading && (
                      <div className="flex justify-start">
                        <div className="max-w-[80%] rounded-2xl px-5 py-3.5 bg-transparent border border-panel-border text-muted-foreground flex items-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin" /> Thinking...
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="p-4 border-t border-border bg-panel">
                    <div className="relative">
                      <input 
                        type="text"
                        value={chatInput}
                        onChange={e => setChatInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleSendChat()}
                        placeholder="Ask a question..."
                        className="w-full bg-panel border border-panel-border text-sm text-primary rounded-xl pl-4 pr-12 py-3.5 outline-none focus:border-zinc-600 transition"
                      />
                      <button 
                        onClick={handleSendChat}
                        disabled={!chatInput.trim() || isChatLoading}
                        className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center bg-accent rounded-lg text-primary disabled:opacity-50 hover:bg-accent"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
          
        </div>
      </main>

      {/* Settings Modal */}
      <AnimatePresence>
        {isSettingsOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-panel border border-border w-full max-w-md rounded-2xl shadow-2xl overflow-hidden"
            >
              <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-panel">
                <h2 className="text-lg font-semibold text-primary">Preferences</h2>
                <button 
                  onClick={() => setIsSettingsOpen(false)}
                  className="p-1 rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="p-6 flex flex-col gap-6 bg-panel">
                
                <div className="flex flex-col gap-2">
                  <label className="text-sm font-medium text-foreground">Gemini API Key</label>
                  <p className="text-xs text-muted-foreground mb-1">Enter your API key to enable extraction and AI chat.</p>
                  <div className="flex gap-2">
                    <input 
                      type="password" 
                      placeholder="AIzaSy..." 
                      value={tempApiKey}
                      onChange={(e) => {
                        setTempApiKey(e.target.value);
                        if (e.target.value !== apiKey) setIsKeyValid(false);
                      }}
                      className="flex-1 bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-zinc-500 font-mono"
                    />
                    <button 
                      onClick={handleConfirmApiKey}
                      className="px-4 py-2 bg-accent text-accent-foreground text-sm font-medium rounded-lg hover:opacity-90 transition-opacity flex items-center gap-2"
                    >
                      {isKeyValid ? <CheckCircle2 className="w-4 h-4 text-green-500" /> : "Confirm"}
                    </button>
                  </div>
                </div>

                <div className="flex flex-col gap-2">
                  <label className="text-sm font-medium text-foreground">Theme</label>
                  <p className="text-xs text-muted-foreground mb-1">Select your theme preference.</p>
                  <select 
                    value={theme}
                    onChange={(e) => changeTheme(e.target.value as "light" | "dark")}
                    className="w-full bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground outline-none focus:border-zinc-500"
                  >
                    <option value="light">☀️ Light</option>
                    <option value="dark">🌙 Dark</option>
                  </select>
                </div>

                <div className="flex flex-col gap-2">
                  <label className="text-sm font-medium text-foreground">Size (Zoom)</label>
                  <p className="text-xs text-muted-foreground mb-1">Adjust the overall scale of the application.</p>
                  <select 
                    value={zoomLevel}
                    onChange={(e) => changeZoom(parseFloat(e.target.value))}
                    className="w-full bg-panel border border-border text-sm text-foreground rounded-lg px-3 py-2.5 outline-none focus:border-zinc-500 transition-colors"
                  >
                    <option value={0.75}>Small (75%)</option>
                    <option value={1}>Default (100%)</option>
                    <option value={1.25}>Big (125%)</option>
                  </select>
                </div>

              </div>
              <div className="bg-muted border-t border-border px-6 py-4 flex justify-end">
                <button 
                  onClick={() => setIsSettingsOpen(false)}
                  className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:opacity-90 transition-opacity"
                >
                  Save & Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
