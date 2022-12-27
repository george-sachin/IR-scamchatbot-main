import axios from 'axios';
const api = axios.create({
    baseURL: "http://35.184.141.203:8000/"
    // baseURL: 'http://localhost:4000/'
})


export const nextResponse = (formObject) => api.post(`/next_response/`, formObject)

export const getAllFormData = token => api.get('/dashboard/formData/', {
    headers: {
        Authorization: 'Bearer ' + token //the token is a variable which holds the token
    }
})

export const reset = token => api.put('/reset/')



const apis = {
    api,
    nextResponse,
    reset
}

export default apis;