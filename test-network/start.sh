./network.sh down
./network.sh up
./network.sh createChannel -c main
./network.sh deployCC -ccp ../manage-scores/manage-scores-chaincode -ccn scoresCC -c main -ccl javascript
./network.sh deployCC -ccp ../model-propose/model-propose-chaincode -ccn modelsCC -c main -ccl javascript