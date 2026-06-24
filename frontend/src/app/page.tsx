"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  UploadCloud, Link as LinkIcon, Loader2, FileText, 
  X, Sparkles, LayoutTemplate, Database, 
  Settings, ChevronRight, CheckCircle2, Download, Clock
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
    <div className="min-h-screen bg-[#000000] text-zinc-300 flex overflow-hidden selection:bg-zinc-800 selection:text-white">
      
      {/* Sidebar */}
      <aside className="w-64 border-r border-zinc-900 bg-[#050505] hidden md:flex flex-col">
        <div className="p-6 flex items-center gap-3">
          <div className="w-8 h-8 rounded-md bg-gradient-to-tr from-zinc-700 to-zinc-900 flex items-center justify-center border border-zinc-700 shadow-lg">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-white tracking-wide text-sm">GoldPan AI</span>
        </div>

        <nav className="flex-1 px-4 py-4 flex flex-col gap-1">
          <div className="px-3 py-2 text-xs font-medium text-zinc-600 uppercase tracking-wider mb-2">Workspace</div>
          <button 
            onClick={() => setActiveView("extractor")}
            className={`flex items-center gap-3 px-3 py-2 text-sm w-full text-left rounded-md transition-colors ${activeView === 'extractor' ? 'bg-zinc-900/50 text-white border border-zinc-800 shadow-inner' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50'}`}
          >
            <LayoutTemplate className="w-4 h-4" />
            Extractor Engine
          </button>
          <button 
            onClick={() => { setActiveView("kb"); fetchKbDocuments(); }}
            className={`flex items-center gap-3 px-3 py-2 text-sm w-full text-left rounded-md transition-colors ${activeView === 'kb' ? 'bg-zinc-900/50 text-white border border-zinc-800 shadow-inner' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50'}`}
          >
            <Database className="w-4 h-4" />
            Knowledge Base
          </button>
        </nav>

        <div className="p-4 border-t border-zinc-900">
          <div className="mb-4">
            <label className="text-xs text-zinc-500 uppercase tracking-wider font-semibold block mb-2">Gemini API Key</label>
            <input 
              type="password" 
              placeholder="AIzaSy..." 
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-zinc-800 text-zinc-300 text-xs rounded-md px-3 py-2.5 outline-none focus:border-zinc-600 focus:ring-1 focus:ring-zinc-600 transition-all font-mono"
            />
          </div>
          <a href="#" className="flex items-center gap-3 px-3 py-2 text-sm text-zinc-500 hover:text-zinc-300 rounded-md transition-colors">
            <Settings className="w-4 h-4" />
            Settings
          </a>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative h-screen overflow-hidden">
        {/* Glow Effect */}
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-zinc-800/20 blur-[120px] pointer-events-none" />

        <header className="h-16 border-b border-zinc-900/50 flex items-center px-8 bg-[#000000]/80 backdrop-blur-md z-10">
          <div className="flex items-center gap-2 text-sm text-zinc-500">
            <span>Workspace</span>
            <ChevronRight className="w-4 h-4" />
            <span className="text-zinc-300">{activeView === 'extractor' ? 'Data Cleansing Pipeline' : 'Knowledge Base Chat'}</span>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          {activeView === "extractor" ? (
            <>
              {/* Left Panel: Controls */}
              <div className="w-full lg:w-1/3 border-r border-zinc-900/50 bg-[#050505]/30 p-8 overflow-y-auto flex flex-col gap-8 z-10">
                
                <div>
                  <h1 className="text-2xl font-semibold text-white mb-2">Extract Content</h1>
                  <p className="text-sm text-zinc-500 leading-relaxed">
                    Transform noisy PDFs and raw HTML into clean, AI-ready Markdown.
                  </p>
                </div>

                <div className="bg-[#0a0a0a] border border-zinc-800/80 rounded-xl p-1 flex shadow-sm">
                  <button 
                    onClick={() => setActiveTab("file")}
                    className={`flex-1 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 ${activeTab === 'file' ? 'bg-zinc-800 text-white shadow-md border border-zinc-700/50' : 'text-zinc-500 hover:text-zinc-300'}`}
                  >
                    Document
                  </button>
                  <button 
                    onClick={() => setActiveTab("url")}
                    className={`flex-1 py-2.5 text-sm font-medium rounded-lg transition-all duration-200 ${activeTab === 'url' ? 'bg-zinc-800 text-white shadow-md border border-zinc-700/50' : 'text-zinc-500 hover:text-zinc-300'}`}
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
                        className={`relative overflow-hidden border border-dashed rounded-2xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 ${
                          isDragging 
                            ? "border-zinc-500 bg-zinc-900/80 shadow-[0_0_30px_rgba(255,255,255,0.05)]" 
                            : "border-zinc-800 bg-[#0a0a0a] hover:bg-[#0f0f11] hover:border-zinc-700"
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
                            <p className="text-white font-medium text-sm">{file.name}</p>
                            <p className="text-zinc-500 text-xs mt-1.5 font-mono">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                          </motion.div>
                        ) : (
                          <>
                            <div className="w-12 h-12 bg-zinc-900 rounded-full flex items-center justify-center mb-4 border border-zinc-800">
                              <UploadCloud className="w-5 h-5 text-zinc-400" />
                            </div>
                            <p className="text-zinc-300 font-medium text-sm">Click or drag document here</p>
                            <p className="text-zinc-600 text-xs mt-2 font-medium tracking-wide">PDF, DOCX, TXT, JPG, MP3</p>
                          </>
                        )}
                      </div>

                      <button 
                        onClick={processFile}
                        disabled={!file || status === "loading"}
                        className="group relative w-full bg-white text-black font-semibold py-3.5 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center overflow-hidden hover:bg-zinc-200 shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)]"
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
                      <div className="bg-[#0a0a0a] border border-zinc-800 rounded-xl overflow-hidden focus-within:border-zinc-600 focus-within:ring-1 focus-within:ring-zinc-600 transition-all">
                        <div className="flex items-center gap-3 p-4">
                          <LinkIcon className="w-4 h-4 text-zinc-500" />
                          <input 
                            type="url" 
                            placeholder="https://example.com/article" 
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            className="bg-transparent border-none outline-none text-sm text-white w-full placeholder:text-zinc-600 font-mono"
                          />
                        </div>
                      </div>
                      
                      <button 
                        onClick={processUrl}
                        disabled={!url || status === "loading"}
                        className="group relative w-full bg-white text-black font-semibold py-3.5 rounded-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center overflow-hidden hover:bg-zinc-200"
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
          <div className="flex-1 bg-[#050505] p-8 overflow-y-auto relative z-10 flex flex-col">
            <div className="flex-1 border border-zinc-900 bg-[#0a0a0a]/50 rounded-2xl shadow-2xl flex flex-col overflow-hidden backdrop-blur-sm">
              <div className="h-14 border-b border-zinc-900 flex items-center justify-between px-4 bg-[#0a0a0a]">
                <div className="flex items-center gap-4">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-zinc-800"></div>
                    <div className="w-3 h-3 rounded-full bg-zinc-800"></div>
                    <div className="w-3 h-3 rounded-full bg-zinc-800"></div>
                  </div>
                  <div className="w-px h-5 bg-zinc-800 mx-1"></div>
                  <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider font-semibold">Markdown Output</span>
                </div>
                
                {/* Actions: Download & Clear */}
                {result && (
                  <div className="flex items-center gap-3">
                    <button 
                      onClick={saveToKb} 
                      disabled={isSavingKb}
                      className="flex items-center gap-2 text-white hover:text-black hover:bg-white transition-colors text-xs font-semibold bg-blue-900/50 px-3 py-1.5 rounded-md border border-blue-700 disabled:opacity-50"
                    >
                      {isSavingKb ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Database className="w-3.5 h-3.5" />}
                      Save to KB
                    </button>
                    <button 
                      onClick={downloadMarkdown} 
                      className="flex items-center gap-2 text-white hover:text-black hover:bg-white transition-colors text-xs font-semibold bg-zinc-800 px-3 py-1.5 rounded-md border border-zinc-700"
                    >
                      <Download className="w-3.5 h-3.5" />
                      Download MD
                    </button>
                    <button 
                      onClick={reset} 
                      className="text-zinc-500 hover:text-red-400 transition-colors"
                      title="Clear Workspace"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                )}
              </div>

              <div className="p-8 flex-1 overflow-y-auto font-sans">
                {status === "idle" && (
                  <div className="h-full flex flex-col items-center justify-center text-zinc-600 gap-4">
                    <FileText className="w-12 h-12 opacity-20" />
                    <p className="text-sm font-medium tracking-wide">Ready to extract some gold.</p>
                  </div>
                )}
                
                {status === "loading" && (
                  <div className="h-full flex flex-col items-center justify-center gap-8 text-zinc-400 w-full max-w-md mx-auto">
                    
                    <div className="flex flex-col items-center gap-2 relative">
                      <Clock className="w-8 h-8 text-zinc-500 mb-2 animate-pulse" />
                      <div className="text-3xl font-mono text-white font-light tracking-widest">
                        {formatTime(remainingSeconds)}
                      </div>
                      <p className="text-xs text-zinc-500 uppercase tracking-widest">Estimated Time Remaining</p>
                      
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
                      <div className="h-1.5 w-full bg-zinc-900 rounded-full overflow-hidden">
                        <motion.div 
                          className="h-full bg-white rounded-full"
                          initial={{ width: "0%" }}
                          animate={{ width: `${progressPercent}%` }}
                          transition={{ ease: "linear", duration: 1 }}
                        />
                      </div>
                      <div className="flex justify-between text-xs font-mono text-zinc-600">
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
              <div className="w-full lg:w-1/3 border-r border-zinc-900/50 bg-[#050505]/30 p-8 overflow-y-auto flex flex-col gap-8 z-10">
                <div>
                  <h1 className="text-2xl font-semibold text-white mb-2">Knowledge Base</h1>
                  <p className="text-sm text-zinc-500 leading-relaxed">
                    Documents available for AI querying.
                  </p>
                </div>

                <div className="flex flex-col gap-3">
                  {kbDocuments.length === 0 ? (
                    <div className="text-zinc-600 text-sm italic">No documents found. Extract something and click 'Save to KB'.</div>
                  ) : (
                    kbDocuments.map((doc, idx) => (
                      <div key={idx} className="bg-[#0a0a0a] border border-zinc-800 p-4 rounded-xl flex items-start gap-3 hover:border-zinc-700 transition-colors">
                        <FileText className="w-5 h-5 text-zinc-500 mt-0.5 shrink-0" />
                        <div>
                          <p className="text-sm font-medium text-zinc-300 line-clamp-1">{doc.title}</p>
                          <p className="text-xs text-zinc-600 mt-1 line-clamp-1 font-mono">{doc.source}</p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
                
                <button onClick={fetchKbDocuments} className="text-xs text-zinc-500 hover:text-zinc-300 text-left underline underline-offset-2">Refresh List</button>
              </div>

              {/* Right Panel: Chat UI */}
              <div className="flex-1 bg-[#050505] p-8 relative z-10 flex flex-col">
                <div className="flex-1 border border-zinc-900 bg-[#0a0a0a]/50 rounded-2xl shadow-2xl flex flex-col overflow-hidden backdrop-blur-sm">
                  <div className="h-14 border-b border-zinc-900 flex items-center justify-between px-4 bg-[#0a0a0a]">
                    <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider font-semibold">GoldPan AI Assistant</span>
                  </div>

                  <div className="flex-1 p-6 overflow-y-auto flex flex-col gap-6 font-sans">
                    {chatHistory.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-zinc-600 gap-4">
                        <Sparkles className="w-12 h-12 opacity-20" />
                        <p className="text-sm font-medium tracking-wide">Ask anything about your Knowledge Base.</p>
                      </div>
                    ) : (
                      chatHistory.map((msg, idx) => (
                        <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[80%] rounded-2xl px-5 py-3.5 ${msg.role === 'user' ? 'bg-zinc-800 text-zinc-200' : 'bg-transparent border border-zinc-800 text-zinc-300'}`}>
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
                        <div className="max-w-[80%] rounded-2xl px-5 py-3.5 bg-transparent border border-zinc-800 text-zinc-500 flex items-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin" /> Thinking...
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="p-4 border-t border-zinc-900 bg-[#0a0a0a]">
                    <div className="relative">
                      <input 
                        type="text"
                        value={chatInput}
                        onChange={e => setChatInput(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleSendChat()}
                        placeholder="Ask a question..."
                        className="w-full bg-[#111] border border-zinc-800 text-sm text-white rounded-xl pl-4 pr-12 py-3.5 outline-none focus:border-zinc-600 transition-all"
                      />
                      <button 
                        onClick={handleSendChat}
                        disabled={!chatInput.trim() || isChatLoading}
                        className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center bg-zinc-800 rounded-lg text-white disabled:opacity-50 hover:bg-zinc-700"
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
    </div>
  );
}
