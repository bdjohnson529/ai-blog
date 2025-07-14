module Jekyll
  module ReadingTimeFilter
    def reading_time(text)
      words_per_minute = 200
      words = text.split.size
      minutes = (words / words_per_minute).ceil
      
      if minutes == 1
        "1 min read"
      else
        "#{minutes} min read"
      end
    end
  end
end

Liquid::Template.register_filter(Jekyll::ReadingTimeFilter)