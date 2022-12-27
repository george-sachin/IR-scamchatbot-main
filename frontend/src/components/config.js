import { createChatBotMessage } from 'react-chatbot-kit';
import DogPicture from './DogPicture/DogPicture.jsx';


const config = {
  initialMessages:[createChatBotMessage("Hi, I am IR ChatBot")],
  /* initialMessages: [createChatBotMessage(
    "Hi, I am IR ChatBot. Please select the topic of conversation?",
    {
      widget: "dogPicture",
    }
  )], */
  state: {
    gist: '',
    infoBox: '',
  },
  /* widgets: [
    {
      widgetName: 'dogPicture',
      widgetFunc: (props) => <DogPicture {...props} />,
    },
  ], */


};

export default config;