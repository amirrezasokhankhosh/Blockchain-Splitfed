'use strict';

const stringify = require("json-stringify-deterministic");
const sortKeysRecursive = require("sort-keys-recursive");
const {Contract} = require("fabric-contract-api");
const { Mutex } = require("async-mutex");

class ModelPropose extends Contract {

    constructor() {
        super();
        this.mutex = new Mutex();
    }

    async InitModels(ctx, numServers) {
        const modelsInfo = {
            id : "modelsInfo",
            numServers : parseInt(numServers),
            remaining : parseInt(numServers)
        }
        await ctx.stub.putState(modelsInfo.id, Buffer.from(stringify(sortKeysRecursive(modelsInfo))));
    }

    async ReadModel(ctx, id) {
        const modelBytes = await ctx.stub.getState(id);
        if (!modelBytes || modelBytes.length === 0) {
            throw Error(`No model exists with id ${id}.`);
        }
        return modelBytes.toString();
    }

    async UpdateModelsInfo(ctx) {
        const modelsInfoBytes = await ctx.stub.getState("modelsInfo");
        const modelsInfoString = modelsInfoBytes.toString();
        let modelsInfo = JSON.parse(modelsInfoString);
        modelsInfo.remaining = modelsInfo.remaining - 1;
        if (modelsInfo.remaining === 0) {
            modelsInfo.remaining = modelsInfo.numServers;
        }
        await ctx.stub.putState(modelsInfo.id, Buffer.from(JSON.stringify(modelsInfo)));
        return modelsInfo.remaining === modelsInfo.numServers;
    }

    async CreateModel(ctx, id, serverPath, clientsPath) {
        const model = {
            id: id,
            serverPath: serverPath,
            clientsPath: JSON.parse(clientsPath)
        };
        await ctx.stub.putState(id, Buffer.from(stringify(sortKeysRecursive(model))));
        const res = await this.UpdateModelsInfo(ctx);
        return JSON.stringify(res);
    }

    async GetAllModels(ctx) {
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
            if (record.id.startsWith("model_")) {
                allResults.push(record);
            }
            result = await iterator.next();
        }
        return JSON.stringify(allResults);
    }

    async DeleteAllModels(ctx) {
        const modelsString = await this.GetAllModels(ctx);
        const models = JSON.parse(modelsString);
        for (const model of models){
            await ctx.stub.deleteState(model.id);
        }
    }
}

module.exports = ModelPropose;