import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.List;
import java.util.ArrayList;

public class TestProducerAndConsumer{
	public static void main(String[] args){
		System.out.println("test start!");
		Storage storage = new Storage();
		ThreadFactory factor = Executors.defaultThreadFactory();
		
		Thread c1 = factor.newThread(new Consumer(storage,1,3));
		Thread c2 = factor.newThread(new Consumer(storage,2,3));
		Thread c3 = factor.newThread(new Consumer(storage,3,3));
		Thread c4 = factor.newThread(new Consumer(storage,4,3));
		Thread c5 = factor.newThread(new Consumer(storage,5,3));
		
		Thread p1 = factor.newThread(new Producer(storage,1,10));
		Thread p2 = factor.newThread(new Producer(storage,2,10));
		Thread p3 = factor.newThread(new Producer(storage,3,10));
		
		c1.start();
		c2.start();
		c3.start();
		c4.start();
		c5.start();
		
		p1.start();
		p2.start();
		p3.start();
		
	}
}

class Storage {
	private static int MAX_NUM = 100;
	private List<String> list = new ArrayList<String>();
	
	public void produce(int thread,int num){
		System.out.println("produce.isHoldslock1:" + thread + Thread.holdsLock(list));
		synchronized(list){
			while(true){
				System.out.println("produce.isHoldslock2:" + thread + Thread.holdsLock(list));
				if(list.size() + num > MAX_NUM || list.size() > MAX_NUM){
					System.out.println("producer:" + thread + "    list.size=" + list.size());
					System.out.println("producer:" + thread + "    pause produce");
					try {
						list.wait();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}else{
					try {
						Thread.sleep(100);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
					for(int i=0; i<num; i++){
						list.add("member");
					}
					System.out.println("producer:" + thread + "    storage add:" + num + "	current storage:" + list.size());
					list.notifyAll();
					try {
						list.wait();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
	}
	
	public void consume(int thread,int num){
		System.out.println("consume.isHoldslock1:" + thread + Thread.holdsLock(list));
		synchronized(list) {
			while(true) {
				System.out.println("consume.isHoldslock2:" + thread + Thread.holdsLock(list));
				if(list.size() < num || list.size() == 0){
					System.out.println("consumer:" + thread + "    list.size=" + list.size());
					System.out.println("consumer:" + thread + "    pause comsume");
					try {
						list.wait();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}else{
					for(int i = 0; i<num; i++){
						list.remove(0);
					}
					System.out.println("consumer:" + thread + "    storage substrict:" + num + "	current storage:" + list.size());
					list.notifyAll();
					try {
						list.wait();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
	}
}

class Producer implements Runnable {
	Storage storage = null;
	int threadNum;
	int num;
	public Producer(Storage storage,int threadNum,int num){
		this.storage = storage;
		this.threadNum = threadNum;
		this.num = num;
	}
	
	@Override
	public void run(){
		System.out.println("start produce thread:" + threadNum);
		this.produce(threadNum,num);
	}
	
	private void produce(int threadNum,int num){
		System.out.println("produce.isHoldslock:" + Thread.holdsLock(storage));
		storage.produce(threadNum,num);
	}
	
}

class Consumer implements Runnable {
	Storage storage = null;
	int threadNum;
	int num;
	public Consumer(Storage storage,int threadNum,int num){
		this.storage = storage;
		this.threadNum = threadNum;
		this.num = num;
	}
	
	@Override
	public void run(){
		this.consume(threadNum,num);
	}
	
	private void consume(int threadNum,int num){
		System.out.println("consume.isHoldslock:" + Thread.holdsLock(storage));
		storage.consume(threadNum,num);
	}
}