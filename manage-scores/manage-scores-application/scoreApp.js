'use strict';

const { TextDecoder } = require("util");

class ScoresApp {
    constructor() {
        this.utf8decoder = new TextDecoder();
    }

    async initScores(contract) {
        try {
            await (await contract).submitTransaction("InitScores");
            return "Scores ledger is successfully initialized.\n"

        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async readScore(contract, id) {
        try {
            const scoreBytes = await (await contract).evaluateTransaction("ReadScore", id);
            const scoreString = scoreBytes.toString();
            return JSON.parse(scoreString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async updateScore(contract, id, score) {
        try {
            await (await contract).submitTransaction("UpdateScore", id, score);
            return `Node ${id}'s score was successfully updated.`
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async getAllScores(contract) {
        try {
            const scoresBytes = await (await contract).evaluateTransaction("GetAllScores");
            const scoresString = scoresBytes.toString();
            return JSON.parse(scoresString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }
}

module.exports = {
    ScoresApp
}