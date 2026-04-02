import '../styles/globals.css'
import type { AppProps } from 'next/app'
import Head from 'next/head'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>JobShield AI — Detect Fake Jobs Instantly</title>
        <meta name="description" content="AI-powered job scam detector. Paste any job posting and know instantly if it's real or fake." />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>" />
      </Head>
      <Component {...pageProps} />
    </>
  )
}
