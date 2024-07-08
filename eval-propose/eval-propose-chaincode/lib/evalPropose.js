'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");
const { Mutex } = require("async-mutex");

class EvalPropose extends Contract {
    constructor() {
        super();
        this.mutex = new Mutex();
    }

    async InitLedger(ctx) {
        const numServers = 3;
        const evalInfo = {
            id : "evalInfo",
            numServers : numServers,
            remaining : numServers
        }
        await ctx.stub.putState(evalInfo.id, Buffer.from(stringify(sortKeysRecursive(evalInfo))));
    }

    async ReadEval(ctx, id) {
        const evalBytes = await ctx.stub.getState(id);
        if (!evalBytes || evalBytes.length === 0) {
            throw Error(`No evaluation exists with id ${id}.`);
        }
        return evalBytes.toString();
    }

    async UpdateEvalInfo(ctx) {
        const release = await this.mutex.acquire();
        try {
            const evalInfoBytes = await ctx.stub.getState("evalInfo");
            const evalInfoString = evalInfoBytes.toString();
            let evalInfo = JSON.parse(evalInfoString);
            evalInfo.remaining = evalInfo.remaining - 1;
            if (evalInfo.remaining === 0) {
                evalInfo.remaining = evalInfo.numServers;
            }
            await ctx.stub.putState(evalInfo.id, Buffer.from(JSON.stringify(evalInfo)));
            var is_equal = evalInfo.remaining === evalInfo.numServers;
        } finally {
            release();
        }
        return is_equal;
    }

    async CreateEval(ctx, id, scores) {
        const evaluation = {
            id : id,
            scores : JSON.parse(scores)
        }
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(evaluation))));
        const res = await this.UpdateEvalInfo(ctx);
        return JSON.stringify(res);
    }

    async UpdateScores(ctx) {
        const evalsString = await this.GetAllEvals(ctx);
        const evals = JSON.parse(evalsString);
        let board = {};
        for (const evaluation of evals) {
            for (const node in evaluation.scores) {
                let score = evaluation.scores[node]
                if (board[node] == null) {
                    board[node] = [score];
                } else {
                    board[node].push(score);
                }
            }

        }
        for (const node in board) {
            let scores = [...board[node]].sort((a, b) => a - b);
            const half = Math.floor(scores.length / 2);
            const medianScore = scores.length % 2 ? scores[half] : (scores[half - 1] + scores[half]) / 2;
            await ctx.stub.invokeChaincode("scoresCC", ["UpdateScore", node, medianScore.toString()], "main");
        }
    }

    async GetAllEvals(ctx) {
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
            if (record.id.startsWith("eval_")) {
                allResults.push(record);
            }
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }
}

module.exports = EvalPropose;