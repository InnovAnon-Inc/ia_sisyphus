#! /usr/bin/env python
# cython: language_level=3
# distutils: language=c++

""" Syslog """

import asyncio
from functools                               import cached_property
import os
import time
from typing                                  import Iterator
from typing                                  import List

import dotenv
from httpx                                   import AsyncClient
from httpx                                   import Limits
from httpx                                   import ReadTimeout
from httpx                                   import ReadError
from httpx                                   import WriteTimeout
from httpx                                   import WriteError
from httpx                                   import ConnectTimeout
from httpx                                   import ConnectError
from httpx                                   import ProtocolError
#from llama_index.agent.llm_compiler          import LLMCompilerAgentWorker
from llama_index.storage.chat_store.redis    import RedisChatStore
from llama_index.embeddings.ollama           import OllamaEmbedding
from llama_index.core                        import Document
from llama_index.core                        import SimpleDirectoryReader
from llama_index.core                        import Settings
from llama_index.core                        import StorageContext
from llama_index.core                        import VectorStoreIndex
from llama_index.core                        import SummaryIndex
from llama_index.core                        import DocumentSummaryIndex
from llama_index.core                        import SimpleKeywordTableIndex
from llama_index.core                        import PropertyGraphIndex
from llama_index.core                        import KnowledgeGraphIndex
from llama_index.core.agent                  import ReActAgent
from llama_index.core.agent                  import AgentRunner
from llama_index.core.agent                  import StructuredPlannerAgent
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.llms.types        import MessageRole
from llama_index.core.base.llms.types        import ChatMessage
from llama_index.core.chat_engine.types      import ChatMode
from llama_index.core.chat_engine.types      import BaseChatEngine
from llama_index.core.base.embeddings.base   import BaseEmbedding
from llama_index.core.extractors             import TitleExtractor
from llama_index.core.extractors             import SummaryExtractor
from llama_index.core.extractors             import QuestionsAnsweredExtractor
from llama_index.core.extractors             import KeywordExtractor
from llama_index.core.extractors             import BaseExtractor
from llama_index.core.evaluation             import RelevancyEvaluator
from llama_index.core.graph_stores           import SimpleGraphStore
from llama_index.core.indices.base           import BaseIndex
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.indices.property_graph import ImplicitPathExtractor
from llama_index.core.ingestion              import DocstoreStrategy
from llama_index.core.ingestion              import IngestionCache
from llama_index.core.ingestion              import IngestionPipeline
from llama_index.core.llms.llm               import LLM
from llama_index.core.memory                 import ChatMemoryBuffer
from llama_index.core.memory                 import VectorMemory
from llama_index.core.memory.types           import DEFAULT_CHAT_STORE_KEY
from llama_index.core.memory.types           import BaseChatStoreMemory
from llama_index.core.memory.types           import BaseMemory
from llama_index.core.node_parser            import SentenceSplitter
from llama_index.core.node_parser            import HierarchicalNodeParser
from llama_index.core.node_parser            import get_leaf_nodes
from llama_index.core.node_parser            import get_root_nodes
from llama_index.core.postprocessor          import PrevNextNodePostprocessor
from llama_index.core.postprocessor          import AutoPrevNextNodePostprocessor
from llama_index.core.postprocessor          import SentenceEmbeddingOptimizer
from llama_index.core.postprocessor          import LongContextReorder
from llama_index.core.postprocessor          import TimeWeightedPostprocessor
from llama_index.core.query_engine           import RetrieverQueryEngine
from llama_index.core.query_engine           import RetrySourceQueryEngine
from llama_index.core.query_engine           import RetryQueryEngine
from llama_index.core.query_engine           import SubQuestionQueryEngine
from llama_index.core.retrievers             import QueryFusionRetriever
from llama_index.core.retrievers             import AutoMergingRetriever
from llama_index.core.storage.chat_store     import BaseChatStore
from llama_index.core.tools                  import QueryEngineTool
from llama_index.core.tools                  import FunctionTool
#from llama_index.extractors.entity           import EntityExtractor
from llama_index.llms.ollama                 import Ollama
from llama_index.multi_modal_llms.ollama     import OllamaMultiModal
from llama_index.node_parser.topic           import TopicNodeParser
from llama_index.readers.database            import DatabaseReader
#from llama_index.retrievers.bm25             import BM25Retriever
from llama_index.storage.docstore.redis      import RedisDocumentStore
from llama_index.storage.index_store.redis   import RedisIndexStore
from llama_index.storage.kvstore.redis       import RedisKVStore as RedisCache
from llama_index.vector_stores.redis         import RedisVectorStore
#from ollama                                  import AsyncClient
#from ollama                                  import Client
from ollama                                  import pull
from redis                                   import Redis
from redisvl.schema                          import IndexSchema
from retry_async                             import retry
from sqlalchemy.exc                          import OperationalError
from sqlalchemy                              import text
from structlog                               import get_logger

from ia_communicate.main                     import communicate
from ia_communicate.main                     import get_limits
from ia_syslog.main                          import create_tables

logger            = get_logger()

MONITOR_QUERY_FMT:str = '''
WITH rows_to_delete AS (
    SELECT *
    FROM SystemEvents
    ORDER BY id DESC
    LIMIT {limit}
)
DELETE FROM SystemEvents
WHERE id IN (SELECT id FROM rows_to_delete)
RETURNING SystemEvents;
'''
    #WHERE FromHost = '{from_host}'

class SyslogDatabaseReader(DatabaseReader):

	def load_data(self, query: str) -> List[Document]:
		"""Query and load data from the Database, returning a list of Documents.

		Args:
			query (str): Query parameter to filter tables and rows.

		Returns:
			List[Document]: A list of Document objects.
		"""
		documents = []
		with self.sql_database.engine.connect() as connection:
			if query is None:
				raise ValueError("A query parameter is necessary to filter the data")
			else:
				result = connection.execute(text(query))

			for item in result.fetchall():
				# fetch each item
				#doc_str = ", ".join(
				#	[f"{col}: {entry}" for col, entry in zip(result.keys(), item)]
				#)
				_doc_str  = []
				doc_meta = {}
				for col, entry in zip(result.keys(), item):
					if col == 'message':
						_doc_str.append(entry)
					else:
						doc_meta[col] = entry
				#assert len(_doc_str) == 1
				doc_str  = ", ".join(_doc_str)
				documents.append(Document(text=doc_str, metadata=doc_meta))
		return documents

class SisyphusConfig():

	def __init__(
		self,
		dbhost    :str,
		dbport    :int,
		dbuser    :str,
		dbpassword:str,
		dbname    :str,
		#from_host :str,
		limit     :int=100,
	)->None:
		super().__init__()
		self.verbose          :bool = True

		self.base_url         :str  = 'http://192.168.2.249:11434'
		self.chat_model       :str  = 'llama3.2'
		self.chat_memory_model:str  = 'llama3.2'
		self.request_timeout  :int  = (60 * 30) # 30 minutes

		self.redis_host       :str  = '192.168.2.249'
		self.redis_port       :int  = 6379
		self.ttl              :int  = self.request_timeout

		self.embed_url        :str  = self.base_url
		self.embed_name       :str  = 'nomic-embed-text'
		self.dims             :int  = 768

		self.dbhost           :str  = dbhost
		self.dbport           :int  = dbport
		self.dbuser           :str  = dbuser
		self.dbpassword       :str  = dbpassword
		self.dbname           :str  = dbname
		#self.dbquery          :str  = MONITOR_QUERY
		#self.from_host        :str = from_host
		self.limit            :int = limit

	@property
	def redis_url(self,)->str:
		return str(f'redis://{self.redis_host}:{self.redis_port}')

	@cached_property
	def redis_client(self,)->Redis:
		return Redis.from_url(self.redis_url,)

	@cached_property
	def chat_llm(self,)->LLM:
		return Ollama(
			base_url       =self.base_url,
			model          =self.chat_model,
			request_timeout=self.request_timeout,
			use_json       =False,
			verbose        =self.verbose,)

	@cached_property
	def chat_memory_llm(self,)->LLM:
		return Ollama(
			base_url       =self.base_url,
			model          =self.chat_memory_model,
			request_timeout=self.request_timeout,
			use_json       =False,
			verbose        =self.verbose,)
		
	@cached_property
	def chat_store(self,)->BaseChatStore:
		return RedisChatStore(
			redis_url     =self.redis_url,
			redis_client  =self.redis_client,
			#aredis_client=self.async_redis_client,
			ttl           =self.ttl,)

	@property
	def chat_store_key(self,)->str:
		return str(f'{DEFAULT_CHAT_STORE_KEY} ({self.namespace})')

	@cached_property
	def memory(self,)->BaseMemory:
		return ChatMemoryBuffer.from_defaults(
			llm       =self.chat_memory_llm,
			chat_store=self.chat_store,)

	@property
	def namespace(self,)->str:
		return 'Sisyphus'
		#return 'Sisyphus ({self.from_host})'

	@property
	def collection(self,)->str:
		return self.namespace

	@property
	def prefix(self,)->str:
		return self.namespace

	@cached_property
	def docstore(self,)->RedisDocumentStore:
		return RedisDocumentStore.from_redis_client(
			redis_client=self.redis_client,
			namespace   =self.namespace,)

	@cached_property
	def index_store(self,)->RedisIndexStore:
		return RedisIndexStore.from_redis_client(
			redis_client=self.redis_client,
			namespace   =self.namespace,
			#collection_suffix=
		)

	@property
	def index_schema_name(self,)->str:
		return str(f'{self.namespace} (vector_store)')

	@cached_property
	def custom_schema(self,)->IndexSchema:
		return IndexSchema.from_dict({
			'index': {'name': self.index_schema_name, 'prefix': self.prefix, },
			'fields': [
				{'type': 'tag',  'name': 'id',},
				{'type': 'tag',  'name': 'doc_id',},
				{'type': 'text', 'name': 'text',},
				{
					'type': 'vector',
					'name': 'vector',
					'attrs': {
						'dims'           : self.dims,
						'algorithm'      : 'hnsw',
						'distance_metric': 'cosine',
					},
				},
			],
		})

	@cached_property
	def vector_store(self,)->RedisVectorStore:
		return RedisVectorStore(
			overwrite   =False,
			redis_client=self.redis_client,
			schema      =self.custom_schema,
			store_text  =True,)

	@cached_property
	def storage_context(self,)->StorageContext:
		return StorageContext.from_defaults(
			docstore   =self.docstore,
			index_store=self.index_store,
			vector_store=self.vector_store,
			# TODO
			# graph_store=
			# image_store=
		)

	@cached_property
	def embed_model(self,)->BaseEmbedding:
		return OllamaEmbedding(
			base_url        =self.embed_url,
			embed_batch_size=1,
			model_name      =self.embed_name,
			request_timeout =self.request_timeout,
			#verbose         =self.verbose,)
		)

	@cached_property
	def index(self,)->BaseIndex:
		return VectorStoreIndex.from_documents(
			[],
			storage_context  =self.storage_context,
			#show_progress    =self.verbose,
			#transformations =self.transformations,
			embed_model      =self.embed_model,
			insert_batch_size=1,)

	@cached_property
	def retriever(self,)->BaseRetriever:
		return self.index.as_retriever(
			similarity_top_k=self.similarity_top_k,)

	@property
	def engine(self,)->BaseQueryEngine:
		return self.index.as_query_engine(
			similarity_top_k=3, # TODO
			# TODO node post processors
		)

	#@property
	#def engine(self,)->BaseChatEngine:
	#	# TODO time-weighted
	#	# TODO node postprocessors
	#	return self.index.as_chat_engine(
	#		llm              =self.chat_llm,
	#		memory           =self.memory,
	#		chat_mode        =ChatMode.CONDENSE_PLUS_CONTEXT,
	#		storage_context  =self.storage_context,
	#		#transformations =self.transformations,
	#		#show_progress    =self.verbose,)
	#	)

	#def chat(self, message:str,)->Iterator[str]:
	#	assert isinstance(message,str), type(message)
	#	response_stream:ChatResponse = self.engine.stream_chat(message,)
	#	for token in response_stream.response_gen:
	#		yield token

	@cached_property
	def reader(self,)->DatabaseReader:
		return SyslogDatabaseReader(
		    scheme  ='postgresql',
		    host    =self.dbhost,
		    port    =self.dbport,
		    user    =self.dbuser,
		    password=self.dbpassword,
		    dbname  =self.dbname,)

	@property
	def dbquery(self,)->str:
		return MONITOR_QUERY_FMT.format(
			#from_host=self.from_host,
			limit=self.limit,)	

	@retry((
        	OperationalError,
	), tries=-1, delay=1, backoff=2, max_delay=None, is_async=False)
	def load_data(self,)->List[Document]:
		documents      :List[Document]         = self.reader.load_data(
	    		#num_workers=
	    		query=self.dbquery,)
		logger.info('#documents      : %s', len(documents),)
		assert documents
		return documents

	def update_index(self,)->None:
		docs:List[Document] = self.load_data()
		self.index.refresh(docs)

#async def communicate(client:AsyncClient, url:str, message:str,)->str:
#	params  :Dict[str,str] = {
#		'message': message,
#		'client' : 'Sisyphus',
#	}
#	#response               = await client.get(url, params=params, timeout=None,) # TODO 600 ?
#	response               = await client.get(url, params=params,)
#	if (response.status_code != 200):
#		await logger.awarn('status code: %s', response.status_code,)
#		return None
#	content :bytes         = response.content
#	result  :str           = content.decode('utf-8')
#	await logger.ainfo('response: %s', result,)
#	return result

async def _main(
	dbhost    :str,
	dbport    :int,
	dbuser    :str,
	dbpassword:str,
	dbname    :str,
	url       :str,
)->None:

	config:SisyphusConfig = SisyphusConfig(
		dbhost    =dbhost,
		dbport    =dbport,
		dbuser    =dbuser,
		dbpassword=dbpassword,
		dbname    =dbname,
	)

	while True:
		config.update_index()

	# TODO disable chat

	#max_connections          :int    = 10
	#max_keepalive_connections:int    =  5
	#limits                   :Limits = Limits(
	#	max_connections          =max_connections,
	#	max_keepalive_connections=max_keepalive_connections,)
	#limits                   :Limits = get_limits()
	
	#async with AsyncClient(limits=limits, timeout=None,) as client:
	#	message:str         = str(f'I am {config.namespace}, the syslog RAG. I am initiating a conversation with Crow Xi, the Operator.')
	#	msg    :ChatMessage = ChatMessage(
	#		role   =MessageRole.ASSISTANT,
	#		content=message,)
	#	await config.memory.aput(message=msg,)
	#	message:str         = await communicate(client=client, url=url, message=message, uid=config.namespace,)
	#	await logger.ainfo('Crow Xi: %s', message,)
	#	assert isinstance(message,str), type(message)

	#	while True:
	#		config.update_index()
	#		response:Iterator[str] = config.chat(message=message,)
	#		message                = ''.join(response)
	#		await logger.ainfo('Sisyphus: %s', message,)
	#		message                = await communicate(client=client, url=url, message=message, uid=config.namespace,)
	#		await logger.ainfo('Crow Xi: %s', message,)
	#		assert isinstance(message,str), type(message)

def main()->None:

	dotenv.load_dotenv()

	dbhost         :str             =     os.getenv ('PGHOST',      '192.168.2.249')
	dbport         :int             = int(os.getenv ('PGPORT',      '5434'))
	dbuser         :str             =     os.getenv ('PGUSER',      'rsyslog')
	dbpassword     :str             =     os.environ['PGPASSWORD']
	dbname         :str             =     os.getenv ('DBNAME',      'Syslog')
	url            :str             =     os.getenv ('CROWXI',      'http://192.168.2.249:10007/')
	logger.info('db host         : %s', dbhost,)
	logger.info('db port         : %s', dbport,)
	logger.info('db user         : %s', dbuser,)
	#logger.debug('db password     : %s', dbpassword,)
	logger.info('db name         : %s', dbname,)
	logger.info('url             : %s', url,)

	asyncio.run(create_tables(
		user    =dbuser,
		password=dbpassword,
		database=dbname,
		host    =dbhost,
		port    =dbport, ))
	asyncio.run(_main(
		dbhost    =dbhost,
		dbport    =dbport,
		dbuser    =dbuser,
		dbpassword=dbpassword,
		dbname    =dbname,
		url       =url, ))

if __name__ == '__main__':
	main()

__author__:str = 'you.com' # NOQA
