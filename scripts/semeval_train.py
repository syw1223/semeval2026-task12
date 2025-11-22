import os
# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import logging
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
import random

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ImprovedBertForMultipleChoice(nn.Module):
    """BERT多选择分类模型"""
    
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, labels=None):
        batch_size, num_choices = input_ids.shape[0], input_ids.shape[1]
        
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(batch_size, num_choices)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        
        return logits

class AERDataset(Dataset):
    """数据集类"""
    
    def __init__(self, questions, docs_by_topic, tokenizer, max_length=384):
        self.questions = questions
        self.docs_by_topic = docs_by_topic
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        topic_id = question['topic_id']
        
        event = question.get('target_event', question.get('question', ''))
        
        options = [
            question.get('option_A', ''),
            question.get('option_B', ''), 
            question.get('option_C', ''),
            question.get('option_D', 'The information provided is insufficient to determine the cause')
        ]
        
        context = self._get_context(topic_id, event)
        
        input_sequences = []
        for option in options:
            if context and len(context) > 10:
                text = f"{event} [SEP] {context} [SEP] {option}"
            else:
                text = f"{event} [SEP] {option}"
            input_sequences.append(text)
        
        encodings = self.tokenizer(
            input_sequences,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        golden_answer = question.get('golden_answer', 'D')
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        label = label_map.get(golden_answer, 3)
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long),
            'question_id': question.get('id', idx)
        }
    
    def _get_context(self, topic_id, event):
        """获取相关文档内容作为上下文"""
        if topic_id not in self.docs_by_topic:
            return ""
        
        topic_data = self.docs_by_topic[topic_id]
        docs = topic_data.get('docs', [])
        
        if not docs:
            return ""
        
        context_parts = []
        for doc in docs[:2]:
            content = doc.get('content', '')
            if content and len(content) > 0:
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(content)
        
        return " ".join(context_parts)

def collate_fn(batch):
    """批处理函数"""
    batch_size = len(batch)
    
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.questions = []
        self.docs_by_topic = {}
    
    def load_data(self, questions_path: str, docs_path: str):
        logger.info("加载数据...")
        
        # 加载questions
        try:
            with open(questions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            self.questions.append(json.loads(line.strip()))
                        except json.JSONDecodeError as e:
                            logger.warning(f"解析JSON行失败: {e}")
                            continue
            logger.info(f"成功加载 {len(self.questions)} 个问题")
        except Exception as e:
            logger.error(f"加载问题文件失败: {e}")
            return False
        
        # 加载docs
        try:
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
                for item in docs_data:
                    topic_id = item.get("topic_id")
                    if topic_id is not None:
                        self.docs_by_topic[topic_id] = item
            logger.info(f"成功加载 {len(self.docs_by_topic)} 个主题的文档")
        except Exception as e:
            logger.error(f"加载文档文件失败: {e}")
            return False
        
        return True
    
    def split_data(self, train_ratio=0.8, seed=42):
        """分割训练集和验证集"""
        total_size = len(self.questions)
        train_size = int(total_size * train_ratio)
        
        # 设置随机种子确保可重复性
        np.random.seed(seed)
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        dev_indices = indices[train_size:]
        
        train_questions = [self.questions[i] for i in train_indices]
        dev_questions = [self.questions[i] for i in dev_indices]
        
        logger.info(f"训练集: {len(train_questions)} 样本")
        logger.info(f"验证集: {len(dev_questions)} 样本")
        
        return train_questions, dev_questions

class EarlyStoppingTrainer:
    """带早停的训练器"""
    
    def __init__(self, model, tokenizer, train_dataloader, dev_dataloader, 
                 device, learning_rate=2e-5, max_epochs=20, patience=5):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_dataloader) * max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        self.model.to(device)
        
        # 早停相关变量
        self.best_dev_accuracy = 0
        self.epochs_no_improve = 0
        self.best_model_state = None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            loss, logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def evaluate(self, dataloader):
        """评估模型"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                _, logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        return accuracy
    
    def train(self):
        """训练过程，包含早停"""
        logger.info("开始训练（使用早停机制）...")
        logger.info(f"最大训练轮数: {self.max_epochs}, 早停耐心值: {self.patience}")
        
        training_history = {
            'train_loss': [],
            'dev_accuracy': [],
            'best_epoch': 0
        }
        
        for epoch in range(1, self.max_epochs + 1):
            logger.info(f"开始第 {epoch}/{self.max_epochs} 轮训练")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            training_history['train_loss'].append(train_loss)
            logger.info(f"训练损失: {train_loss:.4f}")
            
            # 在验证集上评估
            dev_accuracy = self.evaluate(self.dev_dataloader)
            training_history['dev_accuracy'].append(dev_accuracy)
            logger.info(f"验证集准确率: {dev_accuracy:.4f}")
            
            # 早停逻辑
            if dev_accuracy > self.best_dev_accuracy:
                self.best_dev_accuracy = dev_accuracy
                self.epochs_no_improve = 0
                self.best_model_state = self.model.state_dict().copy()
                training_history['best_epoch'] = epoch
                
                # 保存最佳模型
                model_path = 'outputs/best_model.pth'
                torch.save(self.best_model_state, model_path)
                logger.info(f"保存最佳模型，验证集准确率: {self.best_dev_accuracy:.4f}")
            else:
                self.epochs_no_improve += 1
                logger.info(f"验证集准确率未提升，连续 {self.epochs_no_improve} 轮")
            
            # 检查早停条件
            if self.epochs_no_improve >= self.patience:
                logger.info(f"早停触发！在第 {epoch} 轮停止训练")
                logger.info(f"最佳验证集准确率: {self.best_dev_accuracy:.4f} (第 {training_history['best_epoch']} 轮)")
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("已恢复最佳模型状态")
        
        # 最终评估
        final_dev_accuracy = self.evaluate(self.dev_dataloader)
        logger.info(f"最终验证集准确率: {final_dev_accuracy:.4f}")
        
        return training_history

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    
    # ========== 配置参数 ==========
    TRAIN_FILE = "data/train_questions.jsonl" 
    DOC_FILE = "data/train_docs.json"         
    OUTPUT_DIR = "outputs"               
    BATCH_SIZE = 4                       
    MAX_EPOCHS = 20                      
    PATIENCE = 5                         
    LEARNING_RATE = 2e-5                
    MAX_LENGTH = 384                     
    SEED = 42                           
    TRAIN_RATIO = 0.8                   
    
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 设置镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for file_path in [TRAIN_FILE, DOC_FILE]:
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return
    
    processor = DataProcessor()
    if not processor.load_data(TRAIN_FILE, DOC_FILE):
        logger.error("数据加载失败")
        return
    
    train_questions, dev_questions = processor.split_data(
        train_ratio=TRAIN_RATIO, 
        seed=SEED
    )
    
    logger.info("初始化模型...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.sep_token is None:
        tokenizer.sep_token = '[SEP]'
    
    model = ImprovedBertForMultipleChoice(dropout_rate=0.3)
    
    train_dataset = AERDataset(train_questions, processor.docs_by_topic, tokenizer, MAX_LENGTH)
    dev_dataset = AERDataset(dev_questions, processor.docs_by_topic, tokenizer, MAX_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 训练模型
    trainer = EarlyStoppingTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        device=device,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE
    )
    
    history = trainer.train()
    logger.info("训练完成!")

if __name__ == "__main__":
    main()