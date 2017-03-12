import sys
import time
import datetime

def prog_bar_end():
    sys.stdout.write("\r\n")
    sys.stdout.flush()

    

def prog_bar(EVERYTHING_ZERO_INDEXED, total_samples, total_epochs, batch_size, epoch, batch_count, speed):
    assert EVERYTHING_ZERO_INDEXED == True  #   batch_count & epoch should be ZERO indexed
    total_batches = int((total_samples*1.0)/batch_size)

    # the exact output you're looking for:
    sys.stdout.write('\r')
    sys.stdout.write("[%-60s] %d%%" % ('='*((60*(batch_count+1))/total_batches), (100*(batch_count+1))/total_batches))
    sys.stdout.flush()
    sys.stdout.write(", epoch %d/%d, BatchSize: %d sents, Batch: %d/%d, Speed: %d secs/batch, TimeRemain: %s"% (epoch+1, total_epochs, batch_size, batch_count+1, total_batches, speed, str(datetime.timedelta(seconds=int(speed*(total_batches - (batch_count+1)))))))
    sys.stdout.flush()


if __name__=="__main__":
    TOTAL_EPOCHS = 10
    TOTAL_SAMPLES = 100
    BATCH_SIZE = 1
    for e in range(TOTAL_EPOCHS):
        total_b = int((TOTAL_SAMPLES*1.0)/BATCH_SIZE)
        for batch_count in range(total_b):
        
            start = time.time()
            
            # MAIN CODE
            time.sleep(1)
    
            speed = time.time() - start    # seconds per sample
            
            # print prog_bar
            prog_bar(True, TOTAL_SAMPLES, TOTAL_EPOCHS, BATCH_SIZE, e, batch_count, speed=speed)
        prog_bar_end()
    

