import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ResultCard from './ResultCard'

interface AnalysisResult {
  score: number
  verdict: 'Safe' | 'Uncertain' | 'Scam'
  reasons: string[]
  risk_level: 'low' | 'medium' | 'high'
  is_job_post?: boolean
  invalid_reason?: string
  total_scans?: number
}

const ALLOWED_API_URL = 'http://localhost:8000'
const API_URL = (() => {
  const url = process.env.NEXT_PUBLIC_API_URL || ALLOWED_API_URL
  try {
    const p = new URL(url)
    return ['http:', 'https:'].includes(p.protocol) ? url : ALLOWED_API_URL
  } catch { return ALLOWED_API_URL }
})()

const TEXT_STEPS = [
  { icon: '🔍', text: 'Scanning job description...',       duration: 800  },
  { icon: '🏢', text: 'Investigating company identity...',  duration: 1000 },
  { icon: '💰', text: 'Verifying salary vs role match...',  duration: 900  },
  { icon: '🧠', text: 'Running Groq AI 5-layer analysis...', duration: 1200 },
  { icon: '📊', text: 'Calculating threat score...',        duration: 600  },
]
const IMG_STEPS = [
  { icon: '📸', text: 'Reading job poster image...',        duration: 1200 },
  { icon: '🔤', text: 'Extracting text with AI vision...',  duration: 1500 },
  { icon: '🧠', text: 'Running fraud analysis...',          duration: 1000 },
  { icon: '📊', text: 'Calculating threat score...',        duration: 600  },
]

export default function JobAnalyzer() {
  const [tab,     setTab]     = useState<'text' | 'image'>('text')
  const [text,    setText]    = useState('')
  const [imgFile, setImgFile] = useState<File | null>(null)
  const [imgPrev, setImgPrev] = useState<string | null>(null)
  const [result,  setResult]  = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [step,    setStep]    = useState(-1)
  const [done,    setDone]    = useState<number[]>([])
  const [error,   setError]   = useState('')
  const btnRef  = useRef<HTMLButtonElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const ripple = (e: React.MouseEvent<HTMLButtonElement>) => {
    const btn = btnRef.current; if (!btn) return
    const r = btn.getBoundingClientRect(), s = Math.max(r.width, r.height)
    const el = document.createElement('span')
    el.className = 'ripple'
    el.style.cssText = `width:${s}px;height:${s}px;left:${e.clientX-r.left-s/2}px;top:${e.clientY-r.top-s/2}px`
    btn.appendChild(el); setTimeout(() => el.remove(), 700)
  }

  const runSteps = async (steps: typeof TEXT_STEPS) => {
    setDone([])
    for (let i = 0; i < steps.length; i++) {
      setStep(i)
      await new Promise(r => setTimeout(r, steps[i].duration))
      setDone(p => [...p, i])
    }
  }

  const analyze = async (e: React.MouseEvent<HTMLButtonElement>) => {
    ripple(e)
    if (tab === 'text' && text.trim().length < 20) { setError('Please enter at least 20 characters.'); return }
    if (tab === 'image' && !imgFile) { setError('Please select an image.'); return }
    setLoading(true); setError(''); setResult(null); setStep(-1); setDone([])
    // Trigger live counter on homepage
    if (typeof window !== 'undefined' && (window as any).__startScanCounter) {
      (window as any).__startScanCounter()
    }

    const steps = tab === 'image' ? IMG_STEPS : TEXT_STEPS
    const fetchP = tab === 'text'
      ? fetch(`${API_URL}/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_description: text }) })
      : (() => { const fd = new FormData(); fd.append('file', imgFile!); return fetch(`${API_URL}/analyze-image`, { method: 'POST', body: fd }) })()

    const [, data] = await Promise.all([
      runSteps(steps),
      fetchP.then(async res => {
        if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Analysis failed') }
        return res.json() as Promise<AnalysisResult>
      })
    ]).catch(err => {
      setError(err instanceof Error ? err.message : 'Cannot connect to backend on port 8000.')
      setLoading(false); setStep(-1)
      return [null, null]
    })

    if (data) {
      const r = data as AnalysisResult
      setResult(r)
      // Update real counter on homepage
      if (r.total_scans !== undefined && typeof window !== 'undefined') {
        ;(window as any).__updateScanCounter?.(r.total_scans)
      }
    }
    setLoading(false); setStep(-1)
  }

  const reset = () => { setResult(null); setText(''); setError(''); setDone([]); setImgFile(null); setImgPrev(null) }
  const steps = tab === 'image' ? IMG_STEPS : TEXT_STEPS

  return (
    <div className="glass-card overflow-hidden">

      {/* Card header */}
      <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: 'rgba(139,92,246,0.15)' }}>
        <div className="flex items-center gap-2.5">
          <span className="text-xl" style={{ filter: 'drop-shadow(0 0 8px rgba(139,92,246,0.8))' }}>🛡️</span>
          <span className="font-semibold text-white text-sm tracking-wide">JobShield: Secure Scanning</span>
        </div>
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500/70" />
          <div className="w-3 h-3 rounded-full bg-yellow-500/70" />
          <div className="w-3 h-3 rounded-full bg-green-500/70" />
        </div>
      </div>

      <div className="p-6">
        <AnimatePresence mode="wait">

          {/* ══ INPUT ══ */}
          {!loading && !result && (
            <motion.div key="input"
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.3 }}
            >
              {/* Tabs */}
              <div className="flex gap-2 mb-5 p-1 rounded-xl border" style={{ background: 'rgba(15,10,30,0.6)', borderColor: 'rgba(139,92,246,0.15)' }}>
                {(['text', 'image'] as const).map(t => (
                  <button key={t} onClick={() => { setTab(t); setError('') }}
                    className={`flex-1 py-2.5 rounded-lg text-xs font-semibold tracking-wider uppercase transition-all duration-300 flex items-center justify-center gap-2
                      ${tab === t ? 'tab-active' : 'text-white/30 hover:text-white/60'}`}
                  >
                    <span>{t === 'text' ? '📋' : '📸'}</span>
                    <span>{t === 'text' ? 'Paste Text' : 'Upload Image'}</span>
                  </button>
                ))}
              </div>

              {/* Text input */}
              {tab === 'text' && (
                <>
                  <div className="relative mb-1">
                    <div className="scan-line" />
                    <textarea
                      className="w-full h-44 rounded-xl p-4 text-white/80 placeholder-white/20 text-sm resize-none transition-all duration-300 leading-relaxed border"
                      style={{ background: 'rgba(4,20,40,0.8)', borderColor: 'rgba(59,130,246,0.15)' }}
                      placeholder={`Paste the full job description here...\n\nInclude: company name, salary, contact info, requirements\nfor the most accurate fraud detection.`}
                      value={text} onChange={e => setText(e.target.value)} maxLength={10000}
                    />
                  </div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs" style={{ color: 'rgba(96,165,250,0.3)' }}>{text.length.toLocaleString()} / 10,000</span>
                    {text.length > 0 && (
                      <button onClick={() => setText('')} className="text-xs transition-colors hover:text-red-400" style={{ color: 'rgba(255,255,255,0.2)' }}>✕ Clear</button>
                    )}
                  </div>
                </>
              )}

              {/* Image input */}
              {tab === 'image' && (
                <div
                  onClick={() => fileRef.current?.click()}
                  className="relative rounded-xl p-5 text-center cursor-pointer transition-all duration-300 mb-4 border"
                  style={{
                    background: imgPrev ? 'rgba(29,110,245,0.05)' : 'rgba(4,20,40,0.8)',
                    borderColor: imgPrev ? 'rgba(59,130,246,0.4)' : 'rgba(59,130,246,0.15)',
                    borderStyle: 'dashed',
                  }}
                >
                  {imgPrev ? (
                    <div className="space-y-2">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={imgPrev} alt="preview" className="max-h-44 mx-auto rounded-lg object-contain" />
                      <p className="text-blue-300 text-sm">{imgFile?.name}</p>
                      <p className="text-white/25 text-xs">Click to change</p>
                    </div>
                  ) : (
                    <div className="py-5 space-y-2">
                      <motion.div
                        className="flex justify-center mb-1"
                        animate={{ y: [0,-6,0], scale: [1, 1.08, 1] }}
                        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                      >
                        <span className="text-5xl">📸</span>
                      </motion.div>
                      <p className="text-white/60 font-semibold text-sm">Upload Job Poster Image</p>
                      <p className="text-white/20 text-xs">JPG · PNG · WEBP — max 10MB</p>
                    </div>
                  )}
                  <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp"
                    onChange={e => { const f = e.target.files?.[0]; if (f) { setImgFile(f); setImgPrev(URL.createObjectURL(f)); setError('') } }}
                    className="hidden" />
                </div>
              )}

              {/* Error */}
              <AnimatePresence>
                {error && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="rounded-xl px-4 py-3 text-sm mb-4 flex items-center gap-2 border"
                    style={{ background: 'rgba(239,68,68,0.08)', borderColor: 'rgba(239,68,68,0.3)', color: 'rgba(252,165,165,0.9)' }}
                  >
                    <span>⚠️</span><span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* SCAN button */}
              <motion.button
                ref={btnRef}
                onClick={analyze}
                disabled={tab === 'text' ? text.trim().length < 20 : !imgFile}
                className="scan-btn w-full py-4 rounded-xl text-white text-sm tracking-widest uppercase disabled:opacity-25 disabled:cursor-not-allowed flex items-center justify-center gap-3"
                whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}
              >
                <motion.span animate={{ rotate: [0,15,-15,0] }} transition={{ duration: 2, repeat: Infinity }}>
                  {tab === 'text' ? '🔍' : '📸'}
                </motion.span>
                <span>{tab === 'text' ? 'Scan for Fraud' : 'Scan Image'}</span>
              </motion.button>

              <p className="text-center text-xs mt-3 flex items-center justify-center gap-1" style={{ color: 'rgba(96,165,250,0.2)' }}>
                <span>🔒</span><span>Your data is never stored</span>
              </p>
            </motion.div>
          )}

          {/* ══ LOADING ══ */}
          {loading && (
            <motion.div key="loading"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              exit={{ opacity: 0 }} transition={{ duration: 0.3 }}
              className="py-4"
            >
              <div className="text-center mb-6">
                <div className="inline-flex items-center gap-2 mb-3">
                  {['bg-purple-400', 'bg-violet-400', 'bg-indigo-400'].map((c, i) => (
                    <span key={i} className={`w-3 h-3 ${c} rounded-full dot-${i+1}`} />
                  ))}
                </div>
                <p className="font-bold text-white text-base">AI Analysis in Progress</p>
                <p className="text-white/40 text-xs mt-1">Running deep pattern recognition...</p>
              </div>

              <div className="space-y-2">
                {steps.map((s, i) => {
                  const isDone = done.includes(i), isActive = step === i
                  return (
                    <motion.div key={i}
                      initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.08 }}
                      className={`flex items-center gap-3 px-4 py-3 rounded-xl border transition-all duration-500
                        ${isDone   ? 'border-blue-500/30 bg-blue-500/8' : ''}
                        ${isActive ? 'border-blue-400/40 bg-blue-500/10 step-active' : ''}
                        ${!isDone && !isActive ? 'border-white/5 bg-white/[0.02] opacity-30' : ''}`}
                    >
                      <span className="text-lg w-7 text-center">{s.icon}</span>
                      <span className={`text-sm flex-1 ${isDone ? 'text-blue-300' : isActive ? 'text-white' : 'text-white/30'}`}>{s.text}</span>
                      <div className="w-5 h-5 flex items-center justify-center">
                        {isDone && (
                          <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}
                            className="w-5 h-5 bg-green-500/20 border border-green-500/50 rounded-full flex items-center justify-center"
                          >
                            <span className="text-green-400 text-xs">✓</span>
                          </motion.div>
                        )}
                        {isActive && <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />}
                      </div>
                    </motion.div>
                  )
                })}
              </div>

              <div className="mt-5 rounded-full h-1.5 overflow-hidden" style={{ background: 'rgba(59,130,246,0.1)' }}>
                <motion.div className="h-full rounded-full"
                  style={{ background: 'linear-gradient(90deg, #1d6ef5, #0ea5e9)' }}
                  initial={{ width: '0%' }}
                  animate={{ width: `${(done.length / steps.length) * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
              <p className="text-xs text-center mt-2" style={{ color: 'rgba(96,165,250,0.25)' }}>
                {done.length} / {steps.length} checks complete
              </p>
            </motion.div>
          )}

          {/* ══ RESULT ══ */}
          {result && !loading && (
            <motion.div key="result"
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5, ease: [0.22,1,0.36,1] }}
            >
              {/* Invalid input — not a job post */}
              {result.is_job_post === false ? (
                <div className="text-center py-6 space-y-4">
                  <motion.div
                    initial={{ scale: 0 }} animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 300 }}
                    className="text-5xl"
                  >
                    📋
                  </motion.div>
                  <h3 className="text-white font-bold text-lg">Not a Job Description</h3>
                  <p className="text-white/50 text-sm max-w-xs mx-auto leading-relaxed">
                    This doesn&apos;t look like a job posting. Please paste a real job description that includes job title, company name, salary, and requirements.
                  </p>
                  <div className="space-y-2 text-left">
                    {result.reasons.map((r, i) => (
                      <div key={i} className="text-xs px-3 py-2 rounded-xl border"
                        style={{ background: 'rgba(234,179,8,0.08)', borderColor: 'rgba(234,179,8,0.25)', color: 'rgba(253,224,71,0.85)' }}>
                        {r}
                      </div>
                    ))}
                  </div>
                  <motion.button
                    onClick={reset}
                    whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                    className="scan-btn w-full py-3.5 rounded-xl text-white text-xs font-bold tracking-widest uppercase mt-2"
                  >
                    🔄 Try Again
                  </motion.button>
                </div>
              ) : (
                <ResultCard result={result} onReset={reset} />
              )}
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  )
}
