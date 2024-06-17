'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");

class ManageScores extends Contract {

    async InitScores(ctx) {
        const num_servers = 4;
        const num_clients = 2;
        let num_nodes = num_servers * (num_clients + 1);
        for (let i = 0; i < num_nodes; i++) {
            const node = {
                id: `node_${i}`,
                port: 8000 + i,
                score: i
            };
            await ctx.stub.putState(node.id, Buffer.from(stringify(sortKeysRecursive(node))));
        }
        const roundInfo = {
            id : "roundInfo",
            num_servers : num_servers,
            num_clients : num_clients,
            servers : [],
            clients : {}
        }
        await ctx.stub.putState(roundInfo.id, Buffer.from(stringify(sortKeysRecursive(roundInfo))));
    }

    async AssignNodes(ctx) {
        let roundInfoString = await this.ReadScore(ctx, "roundInfo");
        let roundInfo = JSON.parse(roundInfoString);
        let num_nodes = roundInfo.num_servers * (roundInfo.num_clients + 1);
        roundInfo.servers = [];
        roundInfo.clients = {};

        const scoresString = await this.GetAllScores(ctx);
        const scores = JSON.parse(scoresString);
        scores.sort((a, b) => (a.score < b.score ? 1 : -1));
        let j = 0;
        for (let i = 0 ; i < num_nodes ; i++) {
            if (i < roundInfo.num_servers) {
                roundInfo.clients[scores[i].id] = [];
                roundInfo.servers.push(scores[i]);
            } else {
                let current_server = roundInfo.servers[j].id;
                roundInfo.clients[current_server].push(scores[i]);
                if (roundInfo.clients[current_server].length === roundInfo.num_clients) {
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

    async UpdateScore(ctx, id, score) {
        const nodeString = await this.ReadScore(ctx, id);
        let node = JSON.parse(nodeString);
        node.score += parseFloat(score);
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(node))));
    }

    async GetAllScores(ctx) {
        const allResults = [];
        // range query with empty string for startKey and endKey does an open-ended query of all assets in the chaincode namespace.
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