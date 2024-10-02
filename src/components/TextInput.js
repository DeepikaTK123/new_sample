import React, { useState } from 'react';

const TextInput = ({ onSubmit }) => {
    const [text, setText] = useState('');

    const handleChange = (event) => {
        setText(event.target.value);
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        onSubmit(text);
        setText('');
    };

    return (
        <form onSubmit={handleSubmit}>
            <textarea
                value={text}
                onChange={handleChange}
                placeholder="Enter text here"
                rows="5"
                cols="40"
            />
            <button type="submit">Submit</button>
        </form>
    );
};

export default TextInput;
