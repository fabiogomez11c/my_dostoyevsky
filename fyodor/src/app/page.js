export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-6xl font-bold text-center mb-5 text-gray-700">My Dostoevsky</h1>
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex mb-4">
        <input className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:border-gray-500 shadow-lg" placeholder="Write something and send . . ." />
        <button className="px-4 py-2 ml-2 text-white bg-gray-800 rounded-lg shadow-lg transition ease-in-out bg-blue-500 active:bg-violet-700 hover:-translate-y-1 hover:scale-110 hover:bg-indigo-500 duration-300">Send</button>
      </div>
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex ">
        <p>Output</p>
      </div>
    </main>
  )
}