// new file called DogPicture.jsx
import React, { useEffect, useState } from 'react';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormControl from '@mui/material/FormControl';
import FormLabel from '@mui/material/FormLabel';
import { Button } from '@mui/material';
import { nextResponse, reset } from '../../api/api';

const DogPicture = ({setState}) => {
  const [topic, setTopic] = useState('');
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

  return (
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
  );
};

export default DogPicture;