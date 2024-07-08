'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");

class ManageScores extends Contract {

    async InitScores(ctx) {
        const numServers = 3;
        const numClients = 2;
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
        roundInfo.servers = [];
        roundInfo.clients = {};

        const scoresString = await this.GetAllScores(ctx);
        const scores = JSON.parse(scoresString);
        scores.sort((a, b) => (a.score < b.score ? 1 : -1));
        let j = 0;
        for (let i = 0 ; i < numNodes ; i++) {
            if (i < roundInfo.numServers) {
                roundInfo.clients[scores[i].id] = [];
                roundInfo.servers.push(scores[i]);
            } else {
                let currentServer = roundInfo.servers[j].id;
                roundInfo.clients[currentServer].push(scores[i]);
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
        servers.sort((a, b) => (a.score > b.score ? 1 : -1));
        let winners = [];
        for (let i = 0 ; i < parseInt(k) ; i++) {
            winners.push(servers[i].id);
        }
        return JSON.stringify(winners);
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