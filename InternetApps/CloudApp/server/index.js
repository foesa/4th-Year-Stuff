const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
var AWS = require("aws-sdk");
const app = express()
app.use(cors());
app.use(bodyParser.json());

const locations = require('./routes/api/data');

app.use('/api/data', locations);
AWS.config.getCredentials(function(err) {
    if (err) console.log(err.stack);
    else{
        console.log("Access Key:", AWS.config.credentials.accessKeyId);
    }
    AWS.config.update({region: 'us-east-1'});
    console.log("Region: ", AWS.config.region);
});

const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`Server started on port ${port}`));