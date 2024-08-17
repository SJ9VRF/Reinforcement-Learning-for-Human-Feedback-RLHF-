from training.train import RLHFTrainer

def main():
    trainer = RLHFTrainer()
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    main()
