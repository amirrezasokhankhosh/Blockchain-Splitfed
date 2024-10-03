'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");

class ManageScores extends Contract {

    async InitScores(ctx, numServers, numClients) {
        numServers = parseInt(numServers);
        numClients = parseInt(numClients);
        let numNodes = numServers * (numClients + 1);
        for (let i = 0; i < numNodes; i++) {
            const node = {
                id: `node_${i}`,
                port: 8000 + i,
                score: 0
            };
            await ctx.stub.putState(node.id, Buffer.from(stringify(sortKeysRecursive(node))));
        }
        const roundInfo = {
            id : "roundInfo",
            numServers : numServers,
            numClients : numClients,
            servers : [],
            clients : {}
        }
        await ctx.stub.putState(roundInfo.id, Buffer.from(stringify(sortKeysRecursive(roundInfo))));
    }

    async AssignNodes(ctx) {
        let roundInfoString = await this.ReadScore(ctx, "roundInfo");
        let roundInfo = JSON.parse(roundInfoString);
        let numNodes = roundInfo.numServers * (roundInfo.numClients + 1);
        let prevServers = roundInfo.servers;
        roundInfo.servers = [];
        roundInfo.clients = {};

        const scoresString = await this.GetAllScores(ctx);
        let scores = JSON.parse(scoresString);
        scores.sort((a, b) => (a.score > b.score ? 1 : -1));
        
        let prevScores = [];
        let i = 0;
        let found = false
        while (i < numNodes - prevServers.length) {
            found = false
            for (let j = 0 ; j < prevServers.length ; j++) {
                if (scores[i].id === prevServers[j].id) {
                    let prevScore = scores.splice(i, 1);
                    prevScores.push(prevScore[0]);
                    found = true;
                    break;
                }
            }
            if (!found) {
                i++;
            }
        }

        let j = 0;
        let pointer1 = 0;
        let pointer2 = 0;
        for (let i = 0 ; i < numNodes ; i++) {
            if (i < roundInfo.numServers) {
                roundInfo.clients[scores[i].id] = [];
                roundInfo.servers.push(scores[i]);
                pointer1 = i + 1
            } else {
                let currentServer = roundInfo.servers[j].id;
                if (pointer2 === prevServers.length) {
                    roundInfo.clients[currentServer].push(scores[pointer1]);
                    pointer1++;
                } else if (pointer1 === numNodes) {
                    roundInfo.clients[currentServer].push(scores[pointer2]);
                    pointer2++;
                } else if (scores[pointer1].score <= prevScores[pointer2].score) {
                    roundInfo.clients[currentServer].push(scores[pointer1]);
                    pointer1++;
                } else {
                    roundInfo.clients[currentServer].push(prevScores[pointer2]);
                    pointer2++;
                }
                if (roundInfo.clients[currentServer].length === roundInfo.numClients) {
                    j = j + 1;
                }
            }
        }
        await ctx.stub.putState(roundInfo.id, Buffer.from(stringify(sortKeysRecursive(roundInfo))));
        return JSON.stringify(roundInfo);
    }

    async ReadScore(ctx, id) {
        const nodeBytes = await ctx.stub.getState(id);
        if (!nodeBytes || nodeBytes.length === 0) {
            throw Error(`No node exists with id ${id}.`);
        }
        return nodeBytes.toString();
    }

    async GetServers(ctx) {
        const roundInfoString = await this.ReadScore(ctx, "roundInfo");
        const roundInfo = JSON.parse(roundInfoString);
        return JSON.stringify(roundInfo.servers);
    }

    async UpdateScore(ctx, id, score) {
        const nodeString = await this.ReadScore(ctx, id);
        let node = JSON.parse(nodeString);
        node.score = parseFloat(score);
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(node))));
    }

    async GetServersScores(ctx) {
        const serversString = await this.GetServers(ctx);
        let servers = JSON.parse(serversString);
        let newServers = [];
        for (const server of servers) {
            const serverString = await this.ReadScore(ctx, server.id);
            const newServer = JSON.parse(serverString);
            newServers.push(newServer);
        }
        return JSON.stringify(newServers);
    }

    async SelectWinners(ctx, k) {
        const serversString = await this.GetServersScores(ctx);
        let servers = JSON.parse(serversString);
        const roundInfoString = await this.ReadScore(ctx, "roundInfo");
        const roundInfo = JSON.parse(roundInfoString);
        servers.sort((a, b) => (a.score > b.score ? 1 : -1));
        let winners = [];
        let clients = [];
        for (let i = 0 ; i < parseInt(k) ; i++) {
            winners.push(servers[i].id);
            const winnerClients = roundInfo.clients[servers[i].id];
            for (const client of winnerClients){
                clients.push(client.id);
            }
        }
        const res = {
            servers : winners,
            clients : clients
        }
        return JSON.stringify(res);
    }

    async GetAllScores(ctx) {
        const allResults = [];
        const iterator = await ctx.stub.getStateByRange('', '');
        let result = await iterator.next();
        while (!result.done) {
            const strValue = Buffer.from(result.value.value.toString()).toString('utf8');
            let record;
            try {
                record = JSON.parse(strValue);
            } catch (err) {
                console.log(err);
                record = strValue;
            }
            if (record.id.startsWith("node")) {
                allResults.push(record);
            }
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }
}

module.exports = ManageScores;