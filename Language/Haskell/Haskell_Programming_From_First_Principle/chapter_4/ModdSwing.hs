module MoodSwing where

data Mood = Blah | Woot deriving Show

-- 1. Mood is the type constructor.
-- 2. Blah or Woot
-- 3. ChangedMood :: Mood -> Mood

changeMood :: Mood -> Mood
changeMood Blah = Woot
changeMood _ = Blah