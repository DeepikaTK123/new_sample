import React from 'react';

const PredictionResult = ({ result }) => {
    return (
        <div>
            <h2>Prediction Result:</h2>
            <p>{result}</p>
        </div>
    );
};

export default PredictionResult;
