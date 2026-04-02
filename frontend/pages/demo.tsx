import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import Link from 'next/link'

// WhatsApp-style floating background items — security themed
const BG_ITEMS = [
  // Shields
  { type: 'emoji', content: '🛡️', x: 5,  y: 8,  size: 2.5, delay: 0,   dur: 5 },
  { type: 'emoji', content: '🛡️', x: 88, y: 12, size: 3,   delay: 1,   dur: 6 },
  { type: 'emoji', content: '🛡️', x: 3,  y: 45, size: 2,   delay: 2,   dur: 4.5 },
  { type: 'emoji', content: '🛡️', x: 92, y: 55, size: 2.8, delay: 0.5, dur: 5.5 },
  { type: 'emoji', content: '🛡️', x: 15, y: 85, size: 2.2, delay: 1.5, dur: 5 },
  { type: 'emoji', content: '🛡️', x: 80, y: 80, size: 2,   delay: 2.5, dur: 4 },
  // Locks
  { type: 'emoji', content: '🔒', x: 20, y: 5,  size: 1.8, delay: 0.3, dur: 4.5 },
  { type: 'emoji', content: '🔒', x: 75, y: 3,  size: 2,   delay: 1.2, dur: 5 },
  { type: 'emoji', content: '🔒', x: 8,  y: 70, size: 1.6, delay: 0.8, dur: 4 },
  { type: 'emoji', content: '🔒', x: 94, y: 35, size: 2.2, delay: 1.8, dur: 5.5 },
  { type: 'emoji', content: '🔐', x: 45, y: 2,  size: 1.8, delay: 0.6, dur: 4.5 },
  { type: 'emoji', content: '🔐', x: 60, y: 95, size: 2,   delay: 1.4, dur: 5 },
  // Security related
  { type: 'emoji', content: '🔍', x: 30, y: 90, size: 1.8, delay: 0.9, dur: 4 },
  { type: 'emoji', content: '🔍', x: 70, y: 88, size: 1.6, delay: 2.1, dur: 5 },
  { type: 'emoji', content: '⚠️', x: 12, y: 22, size: 1.6, delay: 1.1, dur: 4.5 },
  { type: 'emoji', content: '⚠️', x: 85, y: 65, size: 1.8, delay: 0.4, dur: 5 },
  { type: 'emoji', content: '✅', x: 50, y: 92, size: 1.8, delay: 1.7, dur: 4 },
  { type: 'emoji', content: '🤖', x: 96, y: 75, size: 2,   delay: 0.7, dur: 5.5 },
  { type: 'emoji', content: '🤖', x: 2,  y: 30, size: 1.8, delay: 2.3, dur: 4.5 },
  { type: 'emoji', content: '💼', x: 25, y: 3,  size: 1.8, delay: 1.3, dur: 5 },
  { type: 'emoji', content: '💼', x: 65, y: 6,  size: 1.6, delay: 0.2, dur: 4 },
  { type: 'emoji', content: '🚨', x: 90, y: 90, size: 1.8, delay: 1.6, dur: 5 },
  { type: 'emoji', content: '🚨', x: 7,  y: 92, size: 1.6, delay: 2.8, dur: 4.5 },
]

// Cycling bg colours
const THEMES = [
  { a: 'rgba(124,58,237,0.3)',  b: 'rgba(79,70,229,0.2)',   c: 'rgba(236,72,153,0.12)' },
  { a: 'rgba(16,185,129,0.2)',  b: 'rgba(124,58,237,0.25)', c: 'rgba(245,158,11,0.12)' },
  { a: 'rgba(239,68,68,0.18)',  b: 'rgba(99,102,241,0.25)', c: 'rgba(20,184,166,0.14)' },
  { a: 'rgba(168,85,247,0.25)', b: 'rgba(34,197,94,0.18)',  c: 'rgba(251,146,60,0.12)' },
  { a: 'rgba(56,189,248,0.2)',  b: 'rgba(167,139,250,0.25)',c: 'rgba(244,114,182,0.14)' },
]

// Better image upload SVG icon
function ImageUploadIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 24 24" fill="none"
      stroke="rgba(139,92,246,0.8)" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round"
    >
      {/* Image frame */}
      <rect x="2" y="3" width="16" height="13" rx="2" />
      {/* Sun */}
      <circle cx="6.5" cy="7.5" r="1.5" />
      {/* Mountains */}
      <polyline points="2 14 7 9 11 13" />
      <polyline points="9 13 13 9 18 14" />
      {/* Plus circle */}
      <circle cx="19" cy="19" r="3" />
      <line x1="19" y1="17" x2="19" y2="21" />
      <line x1="17" y1="19" x2="21" y2="19" />
    </svg>
  )
}

export default function Demo() {
  const [bgIdx, setBgIdx] = useState(0)
  const [cursor, setCursor] = useState({ x: -100, y: -100 })
  const [clicking, setClicking] = useState(false)

  useEffect(() => {
    const t = setInterval(() => setBgIdx(p => (p + 1) % THEMES.length), 3000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    const move = (e: MouseEvent) => setCursor({ x: e.clientX, y: e.clientY })
    const down = () => setClicking(true)
    const up   = () => setClicking(false)
    window.addEventListener('mousemove', move)
    window.addEventListener('mousedown', down)
    window.addEventListener('mouseup', up)
    return () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mousedown', down)
      window.removeEventListener('mouseup', up)
    }
  }, [])

  const c = THEMES[bgIdx]

  return (
    <div className="relative min-h-screen overflow-hidden flex flex-col items-center justify-center px-4"
      style={{ background: '#0f0a1e', cursor: 'none', fontFamily: 'Inter, sans-serif' }}>

      {/* ── Shield cursor ── */}
      <motion.div
        className="fixed z-[9999] pointer-events-none select-none"
        style={{ x: cursor.x - 16, y: cursor.y - 16 }}
        animate={{ scale: clicking ? 0.7 : 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      >
        <motion.span
          className="text-3xl block"
          animate={{ rotate: clicking ? [0, -15, 15, 0] : 0 }}
          transition={{ duration: 0.3 }}
          style={{ filter: 'drop-shadow(0 0 12px rgba(139,92,246,1)) drop-shadow(0 0 4px rgba(255,255,255,0.6))' }}
        >
          🛡️
        </motion.span>
      </motion.div>

      {/* ── Cycling colour bg ── */}
      <motion.div
        key={bgIdx}
        className="fixed inset-0 pointer-events-none"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 2.5 }}
        style={{
          background: `
            radial-gradient(ellipse at 15% 50%, ${c.a} 0%, transparent 55%),
            radial-gradient(ellipse at 85% 20%, ${c.b} 0%, transparent 55%),
            radial-gradient(ellipse at 55% 85%, ${c.c} 0%, transparent 50%)
          `
        }}
      />

      {/* ── Deep orbs ── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-15%] left-[-10%] w-[600px] h-[600px] rounded-full opacity-30 blur-[120px]"
          style={{ background: 'radial-gradient(circle, #4c1d95, transparent)' }} />
        <div className="absolute bottom-[-20%] right-[-10%] w-[700px] h-[700px] rounded-full opacity-25 blur-[130px]"
          style={{ background: 'radial-gradient(circle, #3730a3, transparent)' }} />
      </div>

      {/* ── Grid ── */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.04]"
        style={{
          backgroundImage: 'linear-gradient(rgba(139,92,246,0.8) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.8) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
        }}
      />

      {/* ── Stars ── */}
      <div className="fixed inset-0 pointer-events-none">
        {[...Array(30)].map((_, i) => (
          <div key={i} className="absolute rounded-full bg-white"
            style={{
              width: `${1 + (i % 3) * 0.5}px`,
              height: `${1 + (i % 3) * 0.5}px`,
              left: `${(i * 37 + 11) % 100}%`,
              top: `${(i * 53 + 7) % 100}%`,
              opacity: 0.06 + (i % 5) * 0.04,
            }}
          />
        ))}
      </div>

      {/* ── WhatsApp-style floating security items ── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        {BG_ITEMS.map((item, i) => (
          <motion.div
            key={i}
            className="absolute select-none"
            style={{ left: `${item.x}%`, top: `${item.y}%`, fontSize: `${item.size}rem` }}
            animate={{
              y: [0, -18, 0],
              rotate: [-4, 4, -4],
              opacity: [0.25, 0.5, 0.25],
              scale: [1, 1.1, 1],
            }}
            transition={{
              duration: item.dur,
              delay: item.delay,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          >
            {item.content}
          </motion.div>
        ))}
      </div>

      {/* ── DEMO CONTENT ── */}
      <div className="relative z-10 w-full max-w-lg text-center">

        {/* Demo label */}
        <div className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full border text-xs"
          style={{ background: 'rgba(124,58,237,0.2)', borderColor: 'rgba(139,92,246,0.4)', color: 'rgba(196,181,253,0.9)' }}>
          <span className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
          <span>DEMO PREVIEW — Background + Image Icon</span>
          <span className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
        </div>

        <h2 className="text-white font-black text-3xl mb-2">Background Demo</h2>
        <p className="text-white/40 text-sm mb-8">
          Background cycles every 3s • Shield follows cursor • 23 floating security icons
        </p>

        {/* Image upload icon demo */}
        <div className="rounded-2xl p-8 border mb-6"
          style={{ background: 'rgba(30,20,60,0.6)', borderColor: 'rgba(139,92,246,0.2)', backdropFilter: 'blur(20px)' }}>
          <p className="text-white/50 text-xs uppercase tracking-widest mb-6">New Image Upload Icon</p>

          <motion.div
            className="flex flex-col items-center gap-3 py-8 rounded-xl border-2 border-dashed cursor-pointer"
            style={{ borderColor: 'rgba(139,92,246,0.3)', background: 'rgba(139,92,246,0.05)' }}
            whileHover={{ borderColor: 'rgba(139,92,246,0.6)', background: 'rgba(139,92,246,0.1)' }}
            animate={{ y: [0, -4, 0] }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
          >
            <ImageUploadIcon />
            <p className="text-white/60 font-semibold text-sm">Upload Job Poster Image</p>
            <p className="text-white/25 text-xs">JPG · PNG · WEBP — max 10MB</p>
            <p className="text-purple-400/60 text-xs">AI will read and analyze the image</p>
          </motion.div>
        </div>

        {/* Approve / Reject buttons */}
        <div className="flex gap-3">
          <Link href="/" className="flex-1">
            <motion.button
              whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              className="w-full py-3.5 rounded-xl font-bold text-sm border"
              style={{ background: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.3)', color: 'rgba(252,165,165,0.9)' }}
            >
              ✕ Keep Old UI
            </motion.button>
          </Link>
          <motion.button
            whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
            onClick={() => alert('Great! Tell me to apply these changes to the main UI.')}
            className="flex-1 py-3.5 rounded-xl font-bold text-sm"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #4f46e5)', color: 'white' }}
          >
            ✓ Apply to Main UI
          </motion.button>
        </div>
      </div>
    </div>
  )
}
