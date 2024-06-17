'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");

class ManageScores extends Contract {

    async InitScores(ctx) {
        const num_clients = 12;
        for (let i = 0; i < num_clients; i++) {
            const node = {
                id: `node_${i}`,
                port: 8000 + i,
                score: 0
            };
            await ctx.stub.putState(node.id, Buffer.from(stringify(sortKeysRecursive(node))));
        }
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
            allResults.push(record);
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }
}

module.exports = ManageScores;