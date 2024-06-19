'use strict';

const { ScoresApp } = require("../manage-scores/manage-scores-application");
const scoresApp = new ScoresApp();
const { ModelsApp } = require("../model-propose/model-propose-application");
const modelsApp = new ModelsApp();


const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const jsonParser = bodyParser.json();
const port = 3000;

const crypto = require("crypto");
const grpc = require("@grpc/grpc-js");
const {connect, Contract, Identity, Signer, signers} = require("@hyperledger/fabric-gateway");
const fs = require("fs/promises");
const path = require("path");

const mspId = "Org1MSP";

const cryptoPath = path.resolve(__dirname, '..', 'test-network', 'organizations', 'peerOrganizations', 'org1.example.com');
const keyDirPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'keystore');
const certPath = path.resolve(cryptoPath, 'users', 'User1@org1.example.com', 'msp', 'signcerts', 'User1@org1.example.com-cert.pem');
const tlsCertPath = path.resolve(cryptoPath, 'peers', 'peer0.org1.example.com', 'tls', 'ca.crt');

const peerEndPoint = "localhost:7051";
const peerHostAlias = "peer0.org1.example.com";

const contractScores = InitConnection("main", "scoresCC");
const contractModels = InitConnection("main", "modelsCC");

const axios = require("axios");

async function newGrpcConnection() {
    const tlsRootCert = await fs.readFile(tlsCertPath);
    const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);
    return new grpc.Client(peerEndPoint, tlsCredentials, {
        'grpc.ssl_target_name_override': peerHostAlias,
        'grpc.max_send_message_length' : 100 * 1024 * 1024,
        'grpc.max_receive_message_length' : 100 * 1024 * 1024
    });
}

async function newIdentity() {
    const credentials = await fs.readFile(certPath);
    return { mspId, credentials };
}

async function newSigner() {
    const files = await fs.readdir(keyDirPath);
    const keyPath = path.resolve(keyDirPath, files[0]);
    const privateKeyPem = await fs.readFile(keyPath);
    const privateKey = crypto.createPrivateKey(privateKeyPem);
    return signers.newPrivateKeySigner(privateKey);
}

async function InitConnection(channelName, chaincodeName) {
    /*
    * Returns a contract for a given channel and chaincode.
    * */
    const client = await newGrpcConnection();

    const gateway = connect({
        client,
        identity: await newIdentity(),
        signer: await newSigner(),
        // Default timeouts for different gRPC calls
        evaluateOptions: () => {
            return { deadline: Date.now() + 500000 }; // 5 seconds
        },
        endorseOptions: () => {
            return { deadline: Date.now() + 1500000 }; // 15 seconds
        },
        submitOptions: () => {
            return { deadline: Date.now() + 500000 }; // 5 seconds
        },
        commitStatusOptions: () => {
            return { deadline: Date.now() + 6000000 }; // 1 minute
        },
    });

    const network = gateway.getNetwork(channelName);

    return network.getContract(chaincodeName);
}

async function startEvaluation() {
    const servers = await scoresApp.getServers(contractScores);
    for (const server of servers) {
        await axios.get(`http://localhost:${server.port}/server/models/ready/`);
    }
    console.log("Evaluation started.")
}

app.get('/', (req, res) => {
    res.send("Hello World!.");
});

app.post('/api/scores/ledger/', async (req, res) => {
    const message = await scoresApp.initScores(contractScores);
    res.send(message);
});

app.post('/api/assign/', async (req, res) => {
    const message = await scoresApp.assignNodes(contractScores);
    res.send(message);
});

app.get('/api/score/', jsonParser, async (req, res) => {
    const score = await scoresApp.readScore(contractScores, req.body.id);
    res.send(score);
});

app.get('/api/servers/', async (req, res) => {
    const servers = await scoresApp.getServers(contractScores);
    res.send(servers);
});

app.post('/api/scores/', jsonParser, async (req, res) => {
    for (const name in req.body.scores) {
        await scoresApp.updateScore(contractScores, name, req.body.scores[name].toString());
    }
    res.send("Scores were successfully created.");
})

app.get('/api/scores/', async (req, res) => {
    const scores = await scoresApp.getAllScores(contractScores);
    res.send(scores);
})


// **** MODEL PROPOSE API ****
app.post('/api/models/ledger/', async (req, res) => {
    const message = await modelsApp.initModels(contractModels);
    res.send(message);
});

app.post('/api/model/', jsonParser, async (req, res) => {
    const respond = await modelsApp.createModel(contractModels, req.body.id, req.body.serverPath, JSON.stringify(req.body.clientsPath));
    if (respond) {
        setTimeout(startEvaluation, 1);
    }
    res.send("Model was created successfully.");
});

app.get('/api/model/', jsonParser, async (req, res) => {
    const message = await modelsApp.readModel(contractModels, req.body.id);
    res.send(message);
});

app.get('/api/models/', async (req, res) => {
    const message = await modelsApp.getAllModels(contractModels);
    res.send(message);
});

// **** Admin ****
app.get('/start/', async (req, res) => {
    const assigned = await scoresApp.assignNodes(contractScores);
    for (const server of assigned.servers) {
        axios({
            method: 'post',
            url: `http://localhost:${server.port}/server/`,
            headers: {},
            data: {
                clients: assigned.clients[server.id],
            }
        });
    }
    console.log("Training started.")
    res.send("Training started.")
})

app.listen(port, () => {
    console.log(`Server is listening on localhost:${port}.\n`);
});