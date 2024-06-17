./network.sh down
./network.sh up
./network.sh createChannel -c scores
./network.sh deployCC -ccp ../manage-scores/manage-scores-chaincode -ccn scoresCC -c scores -ccl javascript