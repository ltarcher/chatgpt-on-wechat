# encoding:utf-8

import time

from bot.bot import Bot
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf, load_config
from .langchat_session import LangchatSession
import requests
import re
import json

# Langchat对话模型API
class LangchatBot(Bot):
    def __init__(self):
        super().__init__()
        self.sessions = SessionManager(LangchatSession, model=conf().get("model") or "chatglm3-6b")
        self.args = {
            "model_name": conf().get("model") or "chatglm3-6b",  # 对话模型的名称
            "temperature": conf().get("temperature", 0.7),  # 如果设置，值域须为 [0, 1] 我们推荐 0.3，以达到较合适的效果。
            "stream": False,
            "max_tokens": 0,
            "prompt_name": "default",
            "conversation_id": "",
            "history_len": -1,
            "history": []
        }
        self.api_key = conf().get("langchat_api_key")
        self.base_url = conf().get("langchat_base_url", "http://127.0.0.1:7861")

    def reply(self, query, context=None):
        # acquire reply content
        if context.type == ContextType.TEXT:
            logger.info("[LANGCHAT_AI] query={}".format(query))

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["#清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "#清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "#更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[LANGCHAT_AI] session query={}".format(session.messages))

            model = context.get("langchat_model")
            new_args = self.args.copy()
            if model:
                new_args["model"] = model
            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, session_id)

            reply_content = self.reply_text(session, args=new_args)
            logger.debug(
                "[LANGCHAT_AI] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[LANGCHAT_AI] reply {} used 0 tokens.".format(reply_content))
            return reply
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def reply_text(self, session: LangchatSession, args=None, retry_count=0) -> dict:
        """
        call openai's ChatCompletion to get the answer
        :param session: a conversation session
        :param session_id: session id
        :param retry_count: retry count
        :return: {}
        """
        try:
            headers = {
                "Content-Type": "application/json",
                #"Authorization": "Bearer " + self.api_key
            }
            body = args
            body["history"] = session.messages[:-1]
            body["query"] = session.messages[-1]["content"]
            # logger.debug("[LANGCHAT_AI] response={}".format(response))
            # logger.info("[LANGCHAT_AI] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))
            res = requests.post(
                f"{self.base_url}/chat/chat",
                headers=headers,
                json=body
            )
            if res.status_code == 200:
                content = res.text
                # 使用正则表达式移除第一个"data: "，但不匹配后续的"data: "
                match = re.search(r'data: (\{.*?\})', content, re.DOTALL)

                if match:
                    # 正则表达式匹配到的内容是整个 "data:{...}"
                    # 我们只取括号内的部分，即 JSON 字符串
                    data_str = match.group(1)

                    # 将提取出的字符串转换为 JSON 对象
                    content = json.loads(data_str)
                    return {
                        "total_tokens": 999999,
                        "completion_tokens": 100,
                        "content": content["text"]
                    }
            else:
                response = res.json()
                error = response.get("detail")
                logger.error(f"[LANGCHAT_AI] chat failed, status_code={res.status_code}, "
                             f"msg={error.get('msg')}, type={error.get('type')}")

                result = {"completion_tokens": 0, "content": "提问太快啦，请休息一下再问我吧"}
                need_retry = False
                if res.status_code >= 500:
                    # server error, need retry
                    logger.warn(f"[LANGCHAT_AI] do retry, times={retry_count}")
                    need_retry = retry_count < 2
                elif res.status_code == 401:
                    result["content"] = "授权失败，请检查API Key是否正确"
                elif res.status_code == 429:
                    result["content"] = "请求过于频繁，请稍后再试"
                    need_retry = retry_count < 2
                else:
                    need_retry = False

                if need_retry:
                    time.sleep(3)
                    return self.reply_text(session, args, retry_count + 1)
                else:
                    return result
        except Exception as e:
            logger.exception(e)
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if need_retry:
                return self.reply_text(session, args, retry_count + 1)
            else:
                return result
