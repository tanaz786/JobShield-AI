import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface AnalysisResult {
  score: number
  verdict: 'Safe' | 'Uncertain' | 'Scam'
  reasons: string[]
  risk_level: 'low' | 'medium' | 'high'
}
interface Props { result: AnalysisResult; onReset: () => void }

const CFG = {
  Safe: {
    color: '#22c55e', textCls: 'text-green-400', bg: 'rgba(34,197,94,0.08)',
    border: 'rgba(34,197,94,0.25)', vpCls: 'vp-green',
    label: 'CLEAN VERDICT', sub: 'This job looks legitimate. Safe to apply!',
    icon: '✅', barGrad: 'from-green-500 to-emerald-400',
  },
  Uncertain: {
    color: '#eab308', textCls: 'text-yellow-400', bg: 'rgba(234,179,8,0.08)',
    border: 'rgba(234,179,8,0.25)', vpCls: 'vp-yellow',
    label: 'NEEDS VERIFICATION', sub: 'Limited info available — verify before applying.',
    icon: '⚠️', barGrad: 'from-yellow-500 to-amber-400',
  },
  Scam: {
    color: '#ef4444', textCls: 'text-red-400', bg: 'rgba(239,68,68,0.08)',
    border: 'rgba(239,68,68,0.25)', vpCls: 'vp-red',
    label: 'HIGH RISK DETECTED', sub: 'Do NOT pay fees or share personal info!',
    icon: '🚨', barGrad: 'from-red-500 to-rose-400',
  },
}

function CircularMeter({ score, color }: { score: number; color: string }) {
  const [anim, setAnim] = useState(0)
  const r = 54, circ = 2 * Math.PI * r
  const offset = circ - (anim / 100) * circ
  useEffect(() => { const t = setTimeout(() => setAnim(score), 200); return () => clearTimeout(t) }, [score])
  return (
    <div className="relative w-36 h-36 flex items-center justify-center">
      <svg width="144" height="144" className="-rotate-90">
        <circle cx="72" cy="72" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
        <circle cx="72" cy="72" r={r} fill="none" stroke={color} strokeWidth="10"
          strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={offset}
          className="meter-circle" style={{ filter: `drop-shadow(0 0 10px ${color})` }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-orbitron font-black text-3xl text-white">{score}%</span>
        <span className="text-xs" style={{ color: 'rgba(96,165,250,0.4)' }}>risk score</span>
      </div>
    </div>
  )
}

function ReasonItem({ reason, index }: { reason: string; index: number }) {
  const isRed    = reason.startsWith('🚩') || reason.startsWith('🚨')
  const isYellow = reason.startsWith('⚠️')
  const isGreen  = reason.startsWith('✅')
  const cls = isRed ? 'r-red' : isYellow ? 'r-yellow' : isGreen ? 'r-green' : 'r-info'
  return (
    <motion.li
      initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.5 + index * 0.05 }}
      className={`flex items-start gap-2.5 px-3 py-2.5 rounded-xl text-xs border ${cls} leading-relaxed`}
    >
      <span className="mt-0.5 shrink-0 opacity-60">›</span>
      <span>{reason}</span>
    </motion.li>
  )
}

export default function ResultCard({ result, onReset }: Props) {
  const cfg = CFG[result.verdict]
  const [showAll, setShowAll] = useState(false)
  const visible = showAll ? result.reasons : result.reasons.slice(0, 4)

  return (
    <div className="space-y-4">

      {/* ══ 1. VERDICT ══ */}
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.34,1.56,0.64,1] }}
        className={`rounded-2xl p-5 border ${cfg.vpCls}`}
        style={{ background: cfg.bg, borderColor: cfg.border }}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-white/35 text-xs font-medium tracking-widest uppercase mb-1">
              JobShield: Secure Verdict
            </p>
            <motion.h2
              className={`font-orbitron font-bold text-2xl ${cfg.textCls}`}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
              style={{ textShadow: `0 0 20px ${cfg.color}` }}
            >
              {cfg.label}
            </motion.h2>
            <p className="text-white/40 text-xs mt-1">{cfg.sub}</p>
          </div>
          <motion.span
            className="text-4xl"
            initial={{ scale: 0, rotate: -20 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ delay: 0.3, type: 'spring', stiffness: 300 }}
          >
            {cfg.icon}
          </motion.span>
        </div>
      </motion.div>

      {/* ══ 2. SCORE + METER ══ */}
      <motion.div
        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
        className="rounded-2xl p-5 border flex items-center gap-5"
        style={{ background: 'rgba(4,20,40,0.85)', borderColor: 'rgba(59,130,246,0.12)' }}
      >
        <CircularMeter score={result.score} color={cfg.color} />

        <div className="flex-1 space-y-3">
          <div>
            <p className="text-white/30 text-xs uppercase tracking-widest mb-1">Spam Confidence</p>
            <p className={`font-orbitron font-black text-4xl ${cfg.textCls}`}
              style={{ textShadow: `0 0 20px ${cfg.color}` }}>
              {result.score}%
            </p>
          </div>
          <div className="w-full rounded-full h-2 overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
            <motion.div
              className={`h-full rounded-full bg-gradient-to-r ${cfg.barGrad}`}
              initial={{ width: 0 }}
              animate={{ width: `${result.score}%` }}
              transition={{ duration: 2, ease: [0.4,0,0.2,1], delay: 0.4 }}
              style={{ boxShadow: `0 0 12px ${cfg.color}80` }}
            />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-white/25 text-xs">Risk Level</span>
            <span className={`text-xs font-bold uppercase tracking-widest px-2.5 py-1 rounded-full border ${cfg.textCls}`}
              style={{ background: cfg.bg, borderColor: cfg.border }}>
              {result.risk_level}
            </span>
          </div>
        </div>
      </motion.div>

      {/* ══ 3. ANALYSIS DETAILS ══ */}
      <motion.div
        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}
        className="rounded-2xl p-5 border"
        style={{ background: 'rgba(4,20,40,0.85)', borderColor: 'rgba(59,130,246,0.12)' }}
      >
        <div className="flex items-center gap-2 mb-3">
          <span className="text-blue-400 text-sm">🧠</span>
          <p className="font-orbitron text-xs tracking-widest" style={{ color: 'rgba(96,165,250,0.5)' }}>
            ANALYSIS DETAILS
          </p>
          <div className="flex-1 h-px" style={{ background: 'rgba(59,130,246,0.1)' }} />
          <span className="text-xs px-2 py-0.5 rounded-full border"
            style={{ background: 'rgba(59,130,246,0.05)', borderColor: 'rgba(59,130,246,0.15)', color: 'rgba(96,165,250,0.4)' }}>
            {result.reasons.length} signals
          </span>
        </div>

        <ul className="space-y-1.5">
          {visible.map((r, i) => <ReasonItem key={i} reason={r} index={i} />)}
        </ul>

        {result.reasons.length > 4 && (
          <button onClick={() => setShowAll(!showAll)}
            className="mt-2.5 w-full text-xs py-2 rounded-xl border transition-all"
            style={{ color: 'rgba(96,165,250,0.4)', borderColor: 'rgba(59,130,246,0.1)', background: 'transparent' }}
          >
            {showAll ? '▲ Show less' : `▼ Show ${result.reasons.length - 4} more`}
          </button>
        )}
      </motion.div>

      {/* ══ 4. SAFETY TIPS (Scam only) ══ */}
      <AnimatePresence>
        {result.verdict === 'Scam' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }} transition={{ delay: 0.5 }}
            className="rounded-2xl p-4 border"
            style={{ background: 'rgba(40,4,4,0.6)', borderColor: 'rgba(239,68,68,0.2)' }}
          >
            <p className="text-xs uppercase tracking-widest mb-2.5 flex items-center gap-1.5"
              style={{ color: 'rgba(252,165,165,0.6)' }}>
              <span>🛡️</span> Protect Yourself
            </p>
            {[
              '🚫 Never pay registration, training or joining fees',
              '🔍 Google the company name + "scam" before applying',
              '📞 Call company directly using official website number',
              '🏦 Never share bank details or UPI ID with recruiters',
            ].map((tip, i) => (
              <motion.p key={i} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.07 }}
                className="text-xs py-1.5 px-3 rounded-lg mb-1.5 border"
                style={{ background: 'rgba(239,68,68,0.05)', borderColor: 'rgba(239,68,68,0.12)', color: 'rgba(252,165,165,0.65)' }}
              >
                {tip}
              </motion.p>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ══ 5. BUTTONS ══ */}
      <div className="grid grid-cols-2 gap-3">
        <motion.button
          onClick={onReset}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }}
          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
          className="scan-btn py-3.5 rounded-xl text-white text-xs font-bold tracking-widest uppercase flex items-center justify-center gap-2"
        >
          <span>🔄</span> Check Another Job
        </motion.button>
        <motion.button
          onClick={onReset}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.75 }}
          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
          className="py-3.5 rounded-xl text-xs font-bold tracking-widest uppercase border transition-all flex items-center justify-center gap-2"
          style={{ background: 'rgba(4,20,40,0.6)', borderColor: 'rgba(59,130,246,0.15)', color: 'rgba(96,165,250,0.5)' }}
        >
          <span>📊</span> Analysis Details
        </motion.button>
      </div>
    </div>
  )
}
