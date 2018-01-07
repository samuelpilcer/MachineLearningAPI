curl -i -H "Content-Type: application/json" -X POST -d '{"layers":[13,13,45],"inputs":784,"outputs":10,"description":"Test"}' http://dnsaddress/create
curl -X POST --form "file=@fileaddress.csv" http://dnsaddress/uploadtraining/3
curl -i -H "Content-Type: application/json" -X POST -d '{"description":"Test", "training_file":"static/test.csv", "training_columns":784, "output_column":785, "epochs":1}' http://dnsaddress/train/3
curl http://dnsaddress/save/3
curl http://dnsaddress/download_str/3 -o Test.json
curl http://dnsaddress/download_weights/3 -o Test.h5