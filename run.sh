PORT=8888
TOKEN=Xu89321Jkl
docker stop parties_analysis
docker run --name=parties_analysis -d --rm -p $PORT:8888 -e JUPYTER_TOKEN=$TOKEN -e JUPYTER_ENABLE_LAB=yes -v "${PWD}/analysis":/home/jovyan/work jupyter/datascience-notebook
retries=5
curl http://localhost:$PORT
while [ $? -ne 0 -a $retries -gt 0 ] ; do
   echo "Server is starting, retrying the connection..." 
   sleep 1
   retries=$(( $retries - 1 ))
   curl http://localhost:$PORT
done
if [ $? -ne 0 ] ; then
   echo "Can't start the notebook, check if there are docker problems"
   exit 1
fi
echo "Opening the notebook at http://localhost:$PORT/lab/tree/work/political_parties_analysis.ipynb?token=$TOKEN"
open http://localhost:$PORT/lab/tree/work/political_parties_analysis.ipynb?token=$TOKEN

