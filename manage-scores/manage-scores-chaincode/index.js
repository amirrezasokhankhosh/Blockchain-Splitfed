/*
 * Copyright IBM Corp. All Rights Reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

'use strict';

const manageScores = require('./lib/manageScores');

module.exports.ManageScores = manageScores;
module.exports.contracts = [manageScores];
