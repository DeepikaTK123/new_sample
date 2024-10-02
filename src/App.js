import React, { useState } from 'react';

function App() {
    const [input, setInput] = useState('');
    const [prediction, setPrediction] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(''); // Clear previous error messages
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: input }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            setPrediction(data.prediction);
        } catch (error) {
            setError('Failed to fetch prediction');
            console.error('Error:', error);
        }
    };

    return (
        <div className="App">
            <h1>Stress Predictor</h1>
            <form onSubmit={handleSubmit}>
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    rows="4"
                    cols="50"
                />
                <br />
                <button type="submit">Predict</button>
            </form>
            {prediction && <p>Prediction: {prediction}</p>}
            {error && <p style={{ color: 'red' }}>{error}</p>}
        </div>
    );
}

export default App;
