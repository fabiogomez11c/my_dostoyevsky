"use client";
import React, { useState } from 'react';
import axios from 'axios';

import config from './config.json'

const version = 'live'

export default function Home() {
  const [text, setText] = useState(''); // user input
  const [complete, setComplete] = useState(''); // the complete text: user input + model output
  const [output, setOutput] = useState(false);

  const handleChange = (event) => {
    setText(event.target.value);
  }

  const handleSend = async () => {
    try {
      const response = await axios.post(config[version]["model"] + "/predict", {
        "text": text
      });
      setComplete(response.data.text)
    }
    catch (error) {
      console.log(error)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <h1 className="text-6xl font-bold text-center mb-5 text-gray-700">My Dostoevsky</h1>
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex mb-4">
        <input value={text} onChange={handleChange} className="w-full p-2 border border-gray-300 rounded-lg focus:outline-none focus:border-gray-500 shadow-lg" placeholder="Write something and send . . ." />
        <button onClick={handleSend} className="px-4 py-2 ml-2 text-white bg-gray-800 rounded-lg shadow-lg transition ease-in-out bg-blue-500 active:bg-violet-700 hover:-translate-y-1 hover:scale-110 hover:bg-indigo-500 duration-300">Send</button>
      </div>
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex ">
        <div style={{ whiteSpace: 'pre-line' }}>{complete}</div>
      </div>
    </main>
  )
}