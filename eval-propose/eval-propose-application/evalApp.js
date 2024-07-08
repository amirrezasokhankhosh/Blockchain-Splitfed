'use strict';

const { TextDecoder } = require("util");

class EvalApp {
    constructor() {
        this.utf8decoder = new TextDecoder();
    }

    async initLedger(contract) {
        try {
            await (await contract).submitTransaction("InitLedger");
            return "Evaluation ledger is successfully initialized.\n"

        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async createEval(contract, id, scores) {
        let tries = 4;
        while (tries !== 0) {
            try {
                const resBytes = await (await contract).submitTransaction("CreateEval", id, scores);
                const resString = this.utf8decoder.decode(resBytes);
                return JSON.parse(resString);
            } catch (error) {
                tries = tries - 1;
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }
        console.log("MVCC READ Conflict");
        return false;
    }


    async readEval(contract, id) {
        try {
            const evalBytes = await (await contract).evaluateTransaction("ReadEval", id);
            const evalString = this.utf8decoder.decode(evalBytes);
            return JSON.parse(evalString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }


    async updateScores(contract) {
        try {
            await (await contract).submitTransaction("UpdateScores");
            return "Node scores are successfully updated.\n"
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async getAllEvals(contract) {
        try {
            const evalsBytes = await (await contract).evaluateTransaction("GetAllEvals");
            const evalsString = this.utf8decoder.decode(evalsBytes);
            return JSON.parse(evalsString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }
}

module.exports = {
    EvalApp
}