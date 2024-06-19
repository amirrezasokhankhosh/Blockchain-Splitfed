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

    async assignNodes(contract) {
        try {
            const roundInfoBytes = await (await contract).submitTransaction("AssignNodes");
            const roundInfoString = this.utf8decoder.decode(roundInfoBytes);
            return JSON.parse(roundInfoString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async readScore(contract, id) {
        try {
            const scoreBytes = await (await contract).evaluateTransaction("ReadScore", id);
            const scoreString = this.utf8decoder.decode(scoreBytes);
            return JSON.parse(scoreString);
        } catch (error) {
            console.log(error);
            return error;
        }
    }

    async getServers(contract) {
        try {
            const serversBytes = await (await contract).evaluateTransaction("GetServers");
            const serversString = this.utf8decoder.decode(serversBytes);
            return JSON.parse(serversString);
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
            const scoresString = this.utf8decoder.decode(scoresBytes);
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