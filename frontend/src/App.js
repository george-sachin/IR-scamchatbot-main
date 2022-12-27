
import React from "react";
import Chatbot from "react-chatbot-kit";

import config from "./components/config";
import MessageParser from "./components/MessageParser";
import ActionProvider from "./components/ActionProvider";
import './main.css';


function App() {
  const validateInput = (message) => {
    if (message === '')
      return false;
    else
      return true;

  };
  return (
    <div className="App">
    
      <Chatbot
        config={config}
        messageParser={MessageParser}
        actionProvider={ActionProvider}
        placeholderText="Message"
        validator={validateInput}
        headerText='IR Project Chatbot'
      />
    </div>
  );
}

export default App;