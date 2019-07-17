import java.util.concurrent.ThreadFactory;
import java.util.concurrent.Executors;

public class ThreadTest {
    public static void main(String[] args) {
        System.out.println("Hello,world!");
        
        System.out.println("start sleep 5s");
        try {
        	 Thread.sleep(5000);
        } catch (InterruptedException e) {
        	e.printStackTrace();
        }
        System.out.println("end sleep 5s");
        
        //Thread t1 = new PrintGood();
        //Thread t2 = new PrintNice();
        
        //Thread t1 = new Thread(new PrintGood1());
        //Thread t2 = new Thread(new PrintNice1());
        
        ThreadFactory factory = Executors.defaultThreadFactory();
        SynTest synTest = new SynTest();
        Thread t1 = factory.newThread(new PrintSunliang()); 
        Thread t2 = factory.newThread(new PrintSongge()); 
        
        t1.start();
        t2.start();
    }
}

class SynTest {
	public static void staticSynPrint(String str){
		synchronized(SynTest.class){
			for(int i = 0;i<10;i++){
				try{
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				System.out.println("print:" + str + i );						
			}			
		}

	}
	
	public void synPrint(String str){
		for(int i = 0;i<10;i++){
			try{
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			synchronized(this){
				System.out.println("print:" + str + i );
			}							
		}
	}

	public synchronized void synPrint1(String str){
		for(int i = 0;i<10;i++){
			try{
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("print:" + str + i );
		}
	}
	
	public synchronized void synPrint2(String str){
		for(int i = 0;i<10;i++){
			try{
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.out.println("print:" + str + i );
		}
	}
}

class PrintSunliang implements Runnable {
	SynTest synTest = null;
	public PrintSunliang(){
	}
	public PrintSunliang(SynTest synTest){
		this.synTest = synTest;
	}
	
	public void run(){
		SynTest.staticSynPrint("Sunliang");
	}
}

class PrintSongge implements Runnable {
	SynTest synTest = null;
	public PrintSongge(){
	}
	public PrintSongge(SynTest synTest){
		this.synTest = synTest;
	}

	public void run(){
		SynTest.staticSynPrint("Songge");
	}
}

class PrintGood extends Thread {
	public void run(){
	  for(int i=0;i<10000;i++){
			System.out.println("Good:" + i);
		}
	}
}

class PrintNice extends Thread {
	public void run(){
  	for(int i=0;i<10000;i++){
  		System.out.println("Nice:" + i);
  	}
	}
}

class PrintGood1 implements Runnable {
	public void run(){
	  for(int i=0;i<10;i++){
	  	try {
	  		Thread.sleep(1000);
	  	} catch (InterruptedException e) {
	  		e.printStackTrace();
	  	}
			System.out.println("Good:" + i);
		}
	}
}

class PrintNice1 implements Runnable {
	public void run(){
  	for(int i=0;i<10;i++){
  		try {
  			Thread.sleep(1000);
  		} catch (InterruptedException e ) {
  			e.printStackTrace();
  		}
  		System.out.println("Nice:" + i);
  	}
	}
}

