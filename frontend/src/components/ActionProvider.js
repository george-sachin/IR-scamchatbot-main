import React, {useEffect, useState} from 'react';
import { nextResponse, reset } from '../api/api';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormControl from '@mui/material/FormControl';
import FormLabel from '@mui/material/FormLabel';
import { Button } from '@mui/material';
import config from './config';
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
import Backdrop from '@mui/material/Backdrop';


const ActionProvider = ({ createChatBotMessage, setState, children }) => {
  const [topic, setTopic] = useState('');
  const [open, setOpen] = React.useState(false);

  const handleHello = (message) => {
    let formObject = {
      text: message,
      topic: (topic === '') ? undefined : topic
    }
    let options = {loading:true}
    // const botMessage = createChatBotMessage('', options);
    // setState((prev) => ({
    //   ...prev,
    //   messages: [...prev.messages, botMessage],
    // }));
    setOpen(true);
    nextResponse(formObject).then(response => {
      const data = response;
      console.log(data);
      const botMessage = createChatBotMessage(data.data.response, options);
      setOpen(false);
      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, botMessage]
      }));
    });

    

  };

  const resetTopic = ()=>{
    setTopic('');
    reset()
    setState((prev) => ({
      ...prev,
      messages: [{message: 'Hi, I am IR ChatBot', type: 'bot', loading: false, delay: undefined}],
    }));
  }
  const handleRadio = (radio)=>{
    setTopic(radio.target.value);
  };
  useEffect(() => {
    console.log(config.state);
  }, [])
  
  return (
    <div>
      <FormControl sx={{'padding':'1%'}}>
        {/* <FormLabel id="demo-row-radio-buttons-group-label">Topics</FormLabel> */}
        <RadioGroup row aria-labelledby="demo-row-radio-buttons-group-label" value={topic} name="row-radio-buttons-group" onChange={handleRadio}>
          <FormControlLabel value="" control={<Radio />} label="All topics" />
          <FormControlLabel value="healthcare" control={<Radio />} label="Health care" />
          <FormControlLabel value="environment" control={<Radio />} label="Environment" />
          <FormControlLabel value="politics" control={<Radio />} label="Politics" />
          <FormControlLabel value="technology" control={<Radio />} label="Technology" />
          <FormControlLabel value="education" control={<Radio />} label="Education" />
          <Button variant="contained" onClick={resetTopic}>Reset</Button>
        </RadioGroup>
      </FormControl>
      <Backdrop
            sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
            open={open}
      // onClick={handleCloseBackdrop}
      >
          <CircularProgress color="inherit" />
      </Backdrop>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          actions: {
            handleHello,
          },
        });
      })}
    </div>
  );
};

export default ActionProvider;