# init setup

1. setup virtual env (optional):
```
# create new virtual env called ``llm``
python -m venv llm

# install packages
llm/bin/pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu
llm/bin/pip install flask requests
```

# test locally:

```
# xjiang @ xjiang-fedora-PF3WLXAV in ~/mypro/llm_retrieval_qa on git:master o [10:47:55] C:130
$ ~/mypro/venv/llm/bin/python llm_qa.py
done processing db, count: 26
found 4 similar docs for user query 这里面的数字化机会有哪些？
 根据上文，数字农业的机会包括：农业物联网应用服务、感知数据描述和传感设备基础规范、农业物联网监测设施
   农业遥感、导航和通信卫星应用体系、基于北斗自动导航的农机作业监测技术等。
```

# start server

Remember to set the OpenAI API in os environment var:

```
export OPENAI_API_KEY=<YOUR_API_KEY>
```

```
# run flask server
llm/bin/python main.py
```

and use ``test/post_prepare.sh`` to load the doc, and ``test/post_query.py`` to start query:

```
$ ./post_query.sh 
{"sucess": true, "answer": " 根据上文，数字农业的机会包括：农业物联网应用服务、感知数据描述和传感设备基础规范、农业物联网监测设施、农业遥感、导航和通信卫星应用体系、基于北斗自动导航的农机作业监测技术等。"}
```



