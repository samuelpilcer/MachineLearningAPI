curl -H "token:MY_TOKEN_AUTH" -i -H "Content-Type: application/json" -X POST -d '{"layers":[13,13,45],"inputs":784,"outputs":10,"description":"Test"}' http://dnsaddress/create
curl -H "token:MY_TOKEN_AUTH" -X POST --form "file=@fileaddress.csv" http://dnsaddress/uploadtraining/3
curl -H "token:MY_TOKEN_AUTH" -i -H "Content-Type: application/json" -X POST -d '{"description":"Test", "training_file":"static/test.csv", "training_columns":784, "output_column":785, "epochs":1}' http://dnsaddress/train/3
curl -H "token:MY_TOKEN_AUTH" http://dnsaddress/save/3
curl -H "token:MY_TOKEN_AUTH" http://dnsaddress/download_str/3 -o Test.json
curl -H "token:MY_TOKEN_AUTH" http://dnsaddress/download_weights/3 -o Test.h5
curl -H "token:MY_TOKEN_AUTH" -X POST --form "file=@fileaddress.csv" http://dnsaddress/uploadfile/3
curl -H "token:MY_TOKEN_AUTH" -X GET http://dnsaddress/process_file/3
curl -H "token:MY_TOKEN_AUTH" -X GET http://dnsaddress/download_file_processed/3