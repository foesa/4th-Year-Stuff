const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express()
app.use(cors());
app.use(bodyParser.json());

const locations = require('./routes/api/location');

app.use('/api/locations', locations);


const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`Server started on port ${port}`));

