import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import JobAnalyzer from '../components/JobAnalyzer'

function ScanCounter() {
  const [count, setCount] = useState(0)

  // Load real count from backend on mount
  useEffect(() => {
    fetch('http://localhost:8000/stats')
      .then(r => r.json())
      .then(d => setCount(d.total_scans || 0))
      .catch(() => setCount(0))
  }, [])

  // Called after each scan to update count
  useEffect(() => {
    ;(window as any).__updateScanCounter = (newCount: number) => {
      setCount(newCount)
    }
  }, [])

  return (
    <motion.span key={count} initial={{ scale: 1.4, color: '#a78bfa' }} animate={{ scale: 1, color: 'rgba(167,139,250,0.9)' }} transition={{ duration: 0.35 }} className="font-black text-xl">
      {count}
    </motion.span>
  )
}

const BG_THEMES = [
  { a: 'rgba(124,58,237,0.25)', b: 'rgba(79,70,229,0.18)',   c: 'rgba(236,72,153,0.10)' },
  { a: 'rgba(16,185,129,0.18)', b: 'rgba(124,58,237,0.22)',  c: 'rgba(245,158,11,0.10)' },
  { a: 'rgba(239,68,68,0.15)',  b: 'rgba(99,102,241,0.22)',  c: 'rgba(20,184,166,0.12)' },
  { a: 'rgba(168,85,247,0.22)', b: 'rgba(34,197,94,0.15)',   c: 'rgba(251,146,60,0.10)' },
  { a: 'rgba(56,189,248,0.18)', b: 'rgba(167,139,250,0.22)', c: 'rgba(244,114,182,0.12)' },
]

// Line styles — thin with glow
const L1 = { stroke: 'rgba(147,197,253,0.7)', strokeWidth: '0.8' } as const
const L2 = { stroke: 'rgba(99,179,237,0.55)',  strokeWidth: '0.6' } as const
const D1 = { fill: 'rgba(147,197,253,1)' } as const
const D2 = { fill: 'rgba(99,179,237,0.9)' } as const
const G1 = { filter: 'drop-shadow(0 0 3px rgba(147,197,253,0.8))' }
const G2 = { filter: 'drop-shadow(0 0 2px rgba(99,179,237,0.7))' }
const GD = { filter: 'drop-shadow(0 0 7px rgba(147,197,253,1))' }

function CircuitShield() {
  return (
    <svg width="100%" height="100%" viewBox="0 0 1000 900"
      preserveAspectRatio="xMidYMid meet" fill="none"
      style={{ position: 'absolute', top: 0, left: 0 }}>

      <defs>
        {/* Animated dash — lines appear to move forward like data flowing */}
        <style>{`
          .flow-line {
            stroke-dasharray: 12 8;
            animation: flowAnim 2.5s linear infinite;
          }
          .flow-line-slow {
            stroke-dasharray: 8 12;
            animation: flowAnim 4s linear infinite;
          }
          .flow-line-rev {
            stroke-dasharray: 10 10;
            animation: flowAnimRev 3s linear infinite;
          }
          @keyframes flowAnim {
            from { stroke-dashoffset: 0; }
            to   { stroke-dashoffset: -40; }
          }
          @keyframes flowAnimRev {
            from { stroke-dashoffset: 0; }
            to   { stroke-dashoffset: 40; }
          }
          .dot-pulse {
            animation: dotPulse 2s ease-in-out infinite;
          }
          @keyframes dotPulse {
            0%,100% { opacity: 0.7; r: 4; }
            50%     { opacity: 1;   r: 6; }
          }
        `}</style>
      </defs>

      {/* ── SHIELD (wider, centered at 500) ── */}
      <path d="M500 18 L730 100 L730 400 C730 545 625 618 500 655 C375 618 270 545 270 400 L270 100 Z"
        fill="rgba(29,78,216,0.14)" stroke="rgba(147,197,253,0.92)" strokeWidth="3"
        style={{ filter: 'drop-shadow(0 0 22px rgba(147,197,253,0.8))' }} />
      <path d="M500 48 L706 126 L706 400 C706 528 610 594 500 628 C390 594 294 528 294 400 L294 126 Z"
        fill="rgba(37,99,235,0.09)" stroke="rgba(147,197,253,0.5)" strokeWidth="2" />
      <path d="M500 80 L682 154 L682 400 C682 512 594 570 500 602 C406 570 318 512 318 400 L318 154 Z"
        fill="rgba(59,130,246,0.07)" stroke="rgba(147,197,253,0.28)" strokeWidth="1.5" />

      {/* ── LOCK ── */}
      <rect x="456" y="308" width="88" height="68" rx="10"
        fill="rgba(59,130,246,0.2)" stroke="rgba(255,255,255,0.95)" strokeWidth="3.5"
        style={{ filter: 'drop-shadow(0 0 18px rgba(147,197,253,1))' }} />
      <path d="M468 308 C468 274 532 274 532 308"
        fill="none" stroke="rgba(255,255,255,0.95)" strokeWidth="3.5" strokeLinecap="round"
        style={{ filter: 'drop-shadow(0 0 14px rgba(147,197,253,1))' }} />
      <circle cx="500" cy="334" r="13" fill="rgba(255,255,255,0.95)"
        style={{ filter: 'drop-shadow(0 0 10px rgba(147,197,253,1))' }} />
      <rect x="493" y="346" width="14" height="20" rx="7" fill="rgba(255,255,255,0.95)" />

      {/* ════ LEFT — lines go all the way to x=0 ════ */}
      <line x1="270" y1="185" x2="140" y2="185" {...L1} style={G1} className="flow-line" />
      <line x1="140" y1="185" x2="140" y2="135" {...L1} style={G1} className="flow-line" />
      <line x1="140" y1="135" x2="0"   y2="135" {...L1} style={G1} className="flow-line" />
      <circle cx="0" cy="135" r="5.5" {...D1} style={GD} className="dot-pulse" />

      <line x1="270" y1="225" x2="0"   y2="225" {...L1} style={G1} className="flow-line-slow" />
      <circle cx="0" cy="225" r="5" {...D1} style={GD} className="dot-pulse" />

      <line x1="270" y1="265" x2="120" y2="265" {...L1} style={G1} className="flow-line" />
      <line x1="120" y1="265" x2="120" y2="320" {...L1} style={G1} className="flow-line" />
      <line x1="120" y1="320" x2="0"   y2="320" {...L1} style={G1} className="flow-line" />
      <circle cx="0" cy="320" r="5.5" {...D1} style={GD} className="dot-pulse" />

      <line x1="272" y1="305" x2="60"  y2="305" {...L2} style={G2} className="flow-line-slow" />
      <line x1="60"  y1="305" x2="60"  y2="375" {...L2} style={G2} className="flow-line-slow" />
      <line x1="60"  y1="375" x2="0"   y2="375" {...L2} style={G2} className="flow-line-slow" />
      <circle cx="0" cy="375" r="4.5" {...D2} style={G2} className="dot-pulse" />

      <line x1="272" y1="350" x2="0"   y2="350" {...L1} style={G1} className="flow-line" />
      <circle cx="0" cy="350" r="5" {...D1} style={GD} className="dot-pulse" />

      <line x1="272" y1="395" x2="100" y2="395" {...L1} style={G1} className="flow-line-slow" />
      <line x1="100" y1="395" x2="100" y2="465" {...L1} style={G1} className="flow-line-slow" />
      <line x1="100" y1="465" x2="0"   y2="465" {...L1} style={G1} className="flow-line-slow" />
      <circle cx="0" cy="465" r="5.5" {...D1} style={GD} className="dot-pulse" />

      <line x1="275" y1="440" x2="0"   y2="440" {...L2} style={G2} className="flow-line" />
      <circle cx="0" cy="440" r="4.5" {...D2} style={G2} className="dot-pulse" />

      <line x1="285" y1="480" x2="80"  y2="480" {...L2} style={G2} className="flow-line-slow" />
      <line x1="80"  y1="480" x2="80"  y2="550" {...L2} style={G2} className="flow-line-slow" />
      <line x1="80"  y1="550" x2="0"   y2="550" {...L2} style={G2} className="flow-line-slow" />
      <circle cx="0" cy="550" r="5" {...D2} style={G2} className="dot-pulse" />

      <line x1="300" y1="520" x2="0"   y2="520" {...L2} style={G2} className="flow-line" />
      <circle cx="0" cy="520" r="4" {...D2} style={G2} className="dot-pulse" />

      <line x1="320" y1="560" x2="60"  y2="560" {...L2} style={G2} className="flow-line-slow" />
      <line x1="60"  y1="560" x2="60"  y2="620" {...L2} style={G2} className="flow-line-slow" />
      <line x1="60"  y1="620" x2="0"   y2="620" {...L2} style={G2} className="flow-line-slow" />
      <circle cx="0" cy="620" r="4.5" {...D2} style={G2} className="dot-pulse" />

      {/* ════ RIGHT — lines go all the way to x=1000 ════ */}
      <line x1="730" y1="185" x2="860" y2="185" {...L1} style={G1} className="flow-line-rev" />
      <line x1="860" y1="185" x2="860" y2="135" {...L1} style={G1} className="flow-line-rev" />
      <line x1="860" y1="135" x2="1000" y2="135" {...L1} style={G1} className="flow-line-rev" />
      <circle cx="1000" cy="135" r="5.5" {...D1} style={GD} className="dot-pulse" />

      <line x1="730" y1="225" x2="1000" y2="225" {...L1} style={G1} className="flow-line-rev" />
      <circle cx="1000" cy="225" r="5" {...D1} style={GD} className="dot-pulse" />

      <line x1="730" y1="265" x2="880" y2="265" {...L1} style={G1} className="flow-line-rev" />
      <line x1="880" y1="265" x2="880" y2="320" {...L1} style={G1} className="flow-line-rev" />
      <line x1="880" y1="320" x2="1000" y2="320" {...L1} style={G1} className="flow-line-rev" />
      <circle cx="1000" cy="320" r="5.5" {...D1} style={GD} className="dot-pulse" />

      <line x1="728" y1="305" x2="940" y2="305" {...L2} style={G2} className="flow-line-slow" />
      <line x1="940" y1="305" x2="940" y2="375" {...L2} style={G2} className="flow-line-slow" />
      <line x1="940" y1="375" x2="1000" y2="375" {...L2} style={G2} className="flow-line-slow" />
      <circle cx="1000" cy="375" r="4.5" {...D2} style={G2} className="dot-pulse" />

      <line x1="728" y1="350" x2="1000" y2="350" {...L1} style={G1} className="flow-line-rev" />
      <circle cx="1000" cy="350" r="5" {...D1} style={GD} className="dot-pulse" />

      <line x1="728" y1="395" x2="900" y2="395" {...L1} style={G1} className="flow-line-slow" />
      <line x1="900" y1="395" x2="900" y2="465" {...L1} style={G1} className="flow-line-slow" />
      <line x1="900" y1="465" x2="1000" y2="465" {...L1} style={G1} className="flow-line-slow" />
      <circle cx="1000" cy="465" r="5.5" {...D1} style={GD} className="dot-pulse" />

      <line x1="725" y1="440" x2="1000" y2="440" {...L2} style={G2} className="flow-line-rev" />
      <circle cx="1000" cy="440" r="4.5" {...D2} style={G2} className="dot-pulse" />

      <line x1="715" y1="480" x2="920" y2="480" {...L2} style={G2} className="flow-line-slow" />
      <line x1="920" y1="480" x2="920" y2="550" {...L2} style={G2} className="flow-line-slow" />
      <line x1="920" y1="550" x2="1000" y2="550" {...L2} style={G2} className="flow-line-slow" />
      <circle cx="1000" cy="550" r="5" {...D2} style={G2} className="dot-pulse" />

      <line x1="700" y1="520" x2="1000" y2="520" {...L2} style={G2} className="flow-line-rev" />
      <circle cx="1000" cy="520" r="4" {...D2} style={G2} className="dot-pulse" />

      <line x1="680" y1="560" x2="940" y2="560" {...L2} style={G2} className="flow-line-slow" />
      <line x1="940" y1="560" x2="940" y2="620" {...L2} style={G2} className="flow-line-slow" />
      <line x1="940" y1="620" x2="1000" y2="620" {...L2} style={G2} className="flow-line-slow" />
      <circle cx="1000" cy="620" r="4.5" {...D2} style={G2} className="dot-pulse" />

      {/* ════ TOP ════ */}
      <line x1="500" y1="18"  x2="500" y2="0"   {...L1} style={G1} className="flow-line" />
      <circle cx="500" cy="0" r="5.5" {...D1} style={GD} className="dot-pulse" />
      <line x1="458" y1="28"  x2="458" y2="0"   {...L1} style={G1} className="flow-line-slow" />
      <circle cx="458" cy="0" r="4.5" {...D1} style={GD} className="dot-pulse" />
      <line x1="542" y1="28"  x2="542" y2="0"   {...L1} style={G1} className="flow-line-slow" />
      <circle cx="542" cy="0" r="4.5" {...D1} style={GD} className="dot-pulse" />
      <line x1="416" y1="42"  x2="416" y2="5"   {...L2} style={G2} className="flow-line" />
      <line x1="416" y1="5"   x2="340" y2="5"   {...L2} style={G2} className="flow-line" />
      <circle cx="340" cy="5" r="4" {...D2} style={G2} className="dot-pulse" />
      <line x1="584" y1="42"  x2="584" y2="5"   {...L2} style={G2} className="flow-line" />
      <line x1="584" y1="5"   x2="660" y2="5"   {...L2} style={G2} className="flow-line" />
      <circle cx="660" cy="5" r="4" {...D2} style={G2} className="dot-pulse" />

      {/* ════ BOTTOM ════ */}
      <line x1="500" y1="655" x2="500" y2="900" {...L1} style={G1} className="flow-line-rev" />
      <circle cx="500" cy="895" r="5.5" {...D1} style={GD} className="dot-pulse" />
      <line x1="458" y1="645" x2="458" y2="880" {...L1} style={G1} className="flow-line-slow" />
      <circle cx="458" cy="880" r="4.5" {...D1} style={GD} className="dot-pulse" />
      <line x1="542" y1="645" x2="542" y2="880" {...L1} style={G1} className="flow-line-slow" />
      <circle cx="542" cy="880" r="4.5" {...D1} style={GD} className="dot-pulse" />
      <line x1="416" y1="630" x2="416" y2="860" {...L2} style={G2} className="flow-line-rev" />
      <line x1="416" y1="860" x2="330" y2="860" {...L2} style={G2} className="flow-line-rev" />
      <circle cx="330" cy="860" r="4" {...D2} style={G2} className="dot-pulse" />
      <line x1="584" y1="630" x2="584" y2="860" {...L2} style={G2} className="flow-line-rev" />
      <line x1="584" y1="860" x2="670" y2="860" {...L2} style={G2} className="flow-line-rev" />
      <circle cx="670" cy="860" r="4" {...D2} style={G2} className="dot-pulse" />

      {/* Inner dots */}
      <circle cx="340" cy="185" r="3" fill="rgba(147,197,253,0.5)" />
      <circle cx="660" cy="185" r="3" fill="rgba(147,197,253,0.5)" />
      <circle cx="328" cy="310" r="3" fill="rgba(147,197,253,0.45)" />
      <circle cx="672" cy="310" r="3" fill="rgba(147,197,253,0.45)" />
      <circle cx="332" cy="440" r="3" fill="rgba(147,197,253,0.4)" />
      <circle cx="668" cy="440" r="3" fill="rgba(147,197,253,0.4)" />
    </svg>
  )
}

export default function Home() {
  const [bgIdx, setBgIdx] = useState(0)
  const [cursor, setCursor] = useState({ x: -100, y: -100 })
  const [clicking, setClicking] = useState(false)

  useEffect(() => {
    const t = setInterval(() => setBgIdx(p => (p + 1) % BG_THEMES.length), 3000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    const move = (e: MouseEvent) => setCursor({ x: e.clientX, y: e.clientY })
    const down = () => setClicking(true)
    const up = () => setClicking(false)
    window.addEventListener('mousemove', move)
    window.addEventListener('mousedown', down)
    window.addEventListener('mouseup', up)
    return () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mousedown', down)
      window.removeEventListener('mouseup', up)
    }
  }, [])

  const c = BG_THEMES[bgIdx]

  return (
    <div className="relative min-h-screen overflow-hidden" style={{ background: '#0f0a1e', cursor: 'none' }}>

      {/* Shield cursor */}
      <motion.div className="fixed z-[9999] pointer-events-none select-none" style={{ x: cursor.x - 16, y: cursor.y - 16 }} animate={{ scale: clicking ? 0.7 : 1 }} transition={{ type: 'spring', stiffness: 300, damping: 20 }}>
        <motion.span className="text-3xl block" animate={{ rotate: clicking ? [0, -15, 15, 0] : 0 }} transition={{ duration: 0.3 }} style={{ filter: 'drop-shadow(0 0 10px rgba(139,92,246,0.9)) drop-shadow(0 0 4px rgba(255,255,255,0.5))' }}>
          🛡️
        </motion.span>
      </motion.div>

      {/* Cycling colour bg */}
      <motion.div key={bgIdx} className="fixed inset-0 pointer-events-none" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 2.5 }}
        style={{ background: `radial-gradient(ellipse at 15% 50%, ${c.a} 0%, transparent 55%), radial-gradient(ellipse at 85% 20%, ${c.b} 0%, transparent 55%), radial-gradient(ellipse at 55% 85%, ${c.c} 0%, transparent 50%)` }}
      />

      {/* Deep orbs */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="orb absolute top-[-15%] left-[-10%] w-[600px] h-[600px] rounded-full opacity-30 blur-[120px]" style={{ background: 'radial-gradient(circle, #4c1d95, transparent)' }} />
        <div className="orb-2 absolute bottom-[-20%] right-[-10%] w-[700px] h-[700px] rounded-full opacity-25 blur-[130px]" style={{ background: 'radial-gradient(circle, #3730a3, transparent)' }} />
      </div>

      {/* Grid */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.04]" style={{ backgroundImage: 'linear-gradient(rgba(139,92,246,0.8) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.8) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

      {/* Stars */}
      <div className="fixed inset-0 pointer-events-none">
        {[...Array(25)].map((_, i) => (
          <div key={i} className="absolute rounded-full bg-white" style={{ width: `${1 + (i % 3) * 0.5}px`, height: `${1 + (i % 3) * 0.5}px`, left: `${(i * 37 + 11) % 100}%`, top: `${(i * 53 + 7) % 100}%`, opacity: 0.06 + (i % 5) * 0.04 }} />
        ))}
      </div>

      {/* FULL SCREEN CIRCUIT SHIELD */}
      <div className="fixed inset-0 pointer-events-none z-0" style={{ opacity: 0.22 }}>
        <CircuitShield />
      </div>

      {/* MAIN CONTENT */}
      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center px-4 py-12">

        <motion.div className="text-center mb-8" initial={{ opacity: 0, y: -30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
          <motion.div
            className="inline-flex items-center gap-2 mb-5 px-4 py-2 rounded-full border text-xs"
            style={{ background: 'rgba(124,58,237,0.15)', borderColor: 'rgba(139,92,246,0.3)', color: 'rgba(196,181,253,0.9)' }}
            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}
          >
            <span className="text-base" style={{ filter: 'drop-shadow(0 0 8px rgba(139,92,246,0.8))' }}>🛡️</span>
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
            <span className="font-medium tracking-wide">AI-Powered • Real-time • 3-Layer Detection</span>
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
            <span className="text-base" style={{ filter: 'drop-shadow(0 0 8px rgba(139,92,246,0.8))' }}>🛡️</span>
          </motion.div>

          <motion.h1 className="font-black text-white mb-4" style={{ fontSize: 'clamp(2.5rem, 8vw, 4rem)', letterSpacing: '-0.02em' }} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            Job<span className="shimmer-purple">Shield</span> AI
          </motion.h1>

          <motion.p className="text-white/45 text-base max-w-md mx-auto leading-relaxed" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}>
            Don&apos;t get scammed. Paste any job posting and our AI detects fraud in seconds —{' '}
            <span className="text-purple-400 font-semibold">before you lose money or data.</span>
          </motion.p>
        </motion.div>

        <motion.div className="flex items-center justify-center gap-8 mb-8" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.55 }}>
          {[
            { val: '97%', label: 'ML Accuracy',   sub: 'on test data',  color: 'text-green-400' },
            { val: null,  label: 'Jobs Scanned',  sub: 'and counting',  color: 'text-purple-400' },
            { val: '<2s', label: 'Analysis Time', sub: 'per job post',  color: 'text-blue-400' },
          ].map((s, i) => (
            <div key={i} className="text-center">
              {s.val === null ? <div className={`font-black text-2xl ${s.color}`}><ScanCounter /></div> : <p className={`font-black text-2xl ${s.color}`}>{s.val}</p>}
              <p className="text-white/40 text-xs mt-0.5">{s.label}</p>
              <p className="text-white/20 text-xs">{s.sub}</p>
            </div>
          ))}
        </motion.div>

        <motion.div className="w-full max-w-lg" initial={{ opacity: 0, y: 40, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.7, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}>
          <JobAnalyzer />
        </motion.div>

        <motion.div className="flex items-center justify-center gap-6 mt-6 flex-wrap" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}>
          {[
            { icon: '🔒', text: 'Data never stored' },
            { icon: '⚡', text: 'Instant analysis' },
            { icon: '🤖', text: 'Groq AI + ML' },
            { icon: '🆓', text: 'Free to use' },
          ].map(b => (
            <div key={b.text} className="flex items-center gap-1.5 text-white/20 text-xs">
              <span>{b.icon}</span><span>{b.text}</span>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  )
}
